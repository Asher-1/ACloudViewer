// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "mvs/da3_fusion.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "mvs/image.h"
#include "mvs/mat.h"
#include "mvs/workspace.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/reconstruction_log.h"
#include "util/threading.h"

namespace colmap {
namespace mvs {
namespace {

template <typename T>
float Median(std::vector<T>* elems) {
    CHECK(!elems->empty());
    const size_t mid_idx = elems->size() / 2;
    std::nth_element(elems->begin(), elems->begin() + mid_idx, elems->end());
    if (elems->size() % 2 == 0) {
        const float mid_element1 = static_cast<float>((*elems)[mid_idx]);
        const float mid_element2 = static_cast<float>(
                *std::max_element(elems->begin(), elems->begin() + mid_idx));
        return (mid_element1 + mid_element2) / 2.0f;
    }
    return static_cast<float>((*elems)[mid_idx]);
}

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

struct ViewGeometry {
    int view_idx = 0;
    int image_idx = 0;
    Eigen::Matrix<float, 3, 4, Eigen::RowMajor> P;
    Eigen::Matrix<float, 3, 4, Eigen::RowMajor> inv_P;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> inv_R;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    int width = 0;
    int height = 0;
};

struct ViewData {
    ViewGeometry geom;
    const DepthMap* depth_map = nullptr;
    const NormalMap* normal_map = nullptr;
    const Bitmap* bitmap = nullptr;
};

struct FusionData {
    int view_idx = 0;
    int row = 0;
    int col = 0;
    int traversal_depth = 0;

    FusionData() = default;
    FusionData(int view_idx_in, int row_in, int col_in, int traversal_depth_in)
        : view_idx(view_idx_in),
          row(row_in),
          col(col_in),
          traversal_depth(traversal_depth_in) {}
};

float SampleDepthBilinear(const DepthMap& depth_map, float col, float row,
                          int width, int height) {
    col = std::max(0.0f, std::min(static_cast<float>(width - 1), col));
    row = std::max(0.0f, std::min(static_cast<float>(height - 1), row));
    const int col0 = static_cast<int>(std::floor(col));
    const int row0 = static_cast<int>(std::floor(row));
    const int col1 = std::min(col0 + 1, width - 1);
    const int row1 = std::min(row0 + 1, height - 1);
    const float dx = col - static_cast<float>(col0);
    const float dy = row - static_cast<float>(row0);

    const auto at = [&depth_map](int r, int c) { return depth_map.Get(r, c); };
    const float v00 = at(row0, col0);
    const float v01 = at(row0, col1);
    const float v10 = at(row1, col0);
    const float v11 = at(row1, col1);
    const float v0 = v00 * (1.0f - dx) + v01 * dx;
    const float v1 = v10 * (1.0f - dx) + v11 * dx;
    return v0 * (1.0f - dy) + v1 * dy;
}

Eigen::Vector3f SampleNormalCamBilinear(const NormalMap& normal_map, float col,
                                        float row, int width, int height) {
    col = std::max(0.0f, std::min(static_cast<float>(width - 1), col));
    row = std::max(0.0f, std::min(static_cast<float>(height - 1), row));
    const int col0 = static_cast<int>(std::floor(col));
    const int row0 = static_cast<int>(std::floor(row));
    const int col1 = std::min(col0 + 1, width - 1);
    const int row1 = std::min(row0 + 1, height - 1);
    const float dx = col - static_cast<float>(col0);
    const float dy = row - static_cast<float>(row0);

    const auto at = [&normal_map](int r, int c, int ch) {
        return normal_map.Get(r, c, ch);
    };
    Eigen::Vector3f n(0.0f, 0.0f, 0.0f);
    for (int ch = 0; ch < 3; ++ch) {
        const float v00 = at(row0, col0, ch);
        const float v01 = at(row0, col1, ch);
        const float v10 = at(row1, col0, ch);
        const float v11 = at(row1, col1, ch);
        const float v0 = v00 * (1.0f - dx) + v01 * dx;
        const float v1 = v10 * (1.0f - dx) + v11 * dx;
        n(ch) = v0 * (1.0f - dy) + v1 * dy;
    }
    return n;
}

int CountConsistentViews(const ViewData& ref_view,
                         const Eigen::Vector3f& xyz,
                         const Eigen::Vector3f& ref_normal,
                         const std::vector<ViewData>& views,
                         const std::vector<int>* overlapping_view_indices,
                         float max_squared_reproj_error,
                         float max_depth_error,
                         float max_point_dist,
                         float min_cos_normal_error,
                         int check_num_images,
                         int* consistent_view_mask) {
    int mask = 0;
    int num_other_checked = 0;

    auto should_check_view = [&](int target_idx) {
        if (target_idx == ref_view.geom.view_idx) {
            return true;
        }
        if (overlapping_view_indices == nullptr ||
            overlapping_view_indices->empty()) {
            return true;
        }
        for (const int neighbor_idx : *overlapping_view_indices) {
            if (neighbor_idx == target_idx) {
                return true;
            }
        }
        return false;
    };

    for (const auto& target_view : views) {
        const int target_idx = target_view.geom.view_idx;
        if (!should_check_view(target_idx)) {
            continue;
        }
        if (target_idx == ref_view.geom.view_idx) {
            if (target_idx >= 0 && target_idx < 31) {
                mask |= (1 << target_idx);
            }
            continue;
        }
        if (check_num_images > 0 && num_other_checked >= check_num_images) {
            break;
        }
        ++num_other_checked;

        const Eigen::Vector3f proj =
            target_view.geom.P *
            Eigen::Vector4f(xyz.x(), xyz.y(), xyz.z(), 1.0f);
        if (std::abs(proj.z()) < std::numeric_limits<float>::epsilon()) {
            continue;
        }

        const float col = proj.x() / proj.z();
        const float row = proj.y() / proj.z();
        if (col < 0.0f || row < 0.0f ||
            col >= static_cast<float>(target_view.geom.width) ||
            row >= static_cast<float>(target_view.geom.height)) {
            continue;
        }

        const float sampled_depth = SampleDepthBilinear(
            *target_view.depth_map, col, row, target_view.geom.width,
            target_view.geom.height);
        if (sampled_depth <= 0.0f || !std::isfinite(sampled_depth)) {
            continue;
        }

        const float depth_error = std::abs((proj.z() - sampled_depth) / sampled_depth);
        if (depth_error > max_depth_error) {
            continue;
        }

        const int col_nearest = static_cast<int>(std::lround(col));
        const int row_nearest = static_cast<int>(std::lround(row));
        if (col_nearest < 0 || row_nearest < 0 ||
            col_nearest >= target_view.geom.width ||
            row_nearest >= target_view.geom.height) {
            continue;
        }
        const float col_diff = col - static_cast<float>(col_nearest);
        const float row_diff = row - static_cast<float>(row_nearest);
        if (col_diff * col_diff + row_diff * row_diff > max_squared_reproj_error) {
            continue;
        }

        const Eigen::Vector3f target_normal_cam = SampleNormalCamBilinear(
            *target_view.normal_map, col, row, target_view.geom.width,
            target_view.geom.height);
        const Eigen::Vector3f target_normal =
            target_view.geom.inv_R * target_normal_cam;
        if (ref_normal.dot(target_normal) < min_cos_normal_error) {
            continue;
        }

        const Eigen::Vector3f xyz_target =
            target_view.geom.inv_P *
            Eigen::Vector4f(col * sampled_depth, row * sampled_depth,
                            sampled_depth, 1.0f);
        if ((xyz - xyz_target).norm() > max_point_dist) {
            continue;
        }

        if (target_idx >= 0 && target_idx < 31) {
            mask |= (1 << target_idx);
        }
    }

    if (consistent_view_mask != nullptr) {
        *consistent_view_mask = mask;
    }

    int num_views = 0;
    for (int view_mask = mask; view_mask != 0; view_mask &= view_mask - 1) {
        ++num_views;
    }
    return num_views;
}

std::vector<int> ViewMaskToImageIndices(int view_mask,
                                        const std::vector<ViewData>& views) {
    std::vector<int> image_indices;
    for (const auto& view : views) {
        if (view.geom.view_idx >= 0 && view.geom.view_idx < 31 &&
            (view_mask & (1 << view.geom.view_idx))) {
            image_indices.push_back(view.geom.image_idx);
        }
    }
    std::sort(image_indices.begin(), image_indices.end());
    image_indices.erase(
            std::unique(image_indices.begin(), image_indices.end()),
            image_indices.end());
    return image_indices;
}

struct VoxelAccum {
    int view_mask = 0;
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

    void Add(int consistent_view_mask,
             float px,
             float py,
             float pz,
             float nnx,
             float nny,
             float nnz,
             uint8_t cr,
             uint8_t cg,
             uint8_t cb) {
        x += px;
        y += py;
        z += pz;
        if (std::isfinite(nnx) && std::isfinite(nny) && std::isfinite(nnz)) {
            nx += nnx;
            ny += nny;
            nz += nnz;
        }
        r += cr;
        g += cg;
        b += cb;
        view_mask |= consistent_view_mask;
        ++count;
    }

    int NumViews() const {
        int num = 0;
        for (int mask = view_mask; mask != 0; mask &= mask - 1) {
            ++num;
        }
        return num;
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
        const double nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
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

    std::vector<int> VisibleImageIndices(
            const std::vector<ViewData>& views) const {
        return ViewMaskToImageIndices(view_mask, views);
    }
};

VoxelKey PointToVoxelKey(const Eigen::Vector3f& xyz,
                         double voxel_size,
                         double origin_x,
                         double origin_y,
                         double origin_z) {
    const double ref_x = (static_cast<double>(xyz.x()) - origin_x) / voxel_size;
    const double ref_y = (static_cast<double>(xyz.y()) - origin_y) / voxel_size;
    const double ref_z = (static_cast<double>(xyz.z()) - origin_z) / voxel_size;
    return VoxelKey{static_cast<int>(std::floor(ref_x)),
                    static_cast<int>(std::floor(ref_y)),
                    static_cast<int>(std::floor(ref_z))};
}

bool AccumulateVoxelSample(const std::vector<ViewData>& views,
                           const std::vector<std::vector<int>>& view_overlaps,
                           const DA3FusionOptions& options,
                           double origin_x,
                           double origin_y,
                           double origin_z,
                           int view_idx,
                           int row,
                           int col,
                           std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash>* voxels) {
    const auto& view = views.at(view_idx);
    const float depth = view.depth_map->Get(row, col);
    if (depth <= 0.0f || !std::isfinite(depth)) {
        return false;
    }

    const float max_squared_reproj_error =
        static_cast<float>(options.max_reproj_error * options.max_reproj_error);
    const float max_depth_error = static_cast<float>(options.max_depth_error);
    const float max_point_dist = static_cast<float>(options.max_point_dist);
    const float min_cos_normal_error =
        std::cos(DegToRad(static_cast<float>(options.max_normal_error)));

    const Eigen::Vector3f xyz =
        view.geom.inv_P *
        Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);
    const Eigen::Vector3f normal =
        view.geom.inv_R *
        Eigen::Vector3f(view.normal_map->Get(row, col, 0),
                        view.normal_map->Get(row, col, 1),
                        view.normal_map->Get(row, col, 2));

    int consistent_view_mask = 0;
    const std::vector<int>* overlapping_view_indices = nullptr;
    if (options.restrict_to_overlapping_views &&
        view_idx >= 0 &&
        view_idx < static_cast<int>(view_overlaps.size())) {
        overlapping_view_indices = &view_overlaps[view_idx];
    }
    const int num_consistent_views = CountConsistentViews(
        view, xyz, normal, views, overlapping_view_indices,
        max_squared_reproj_error, max_depth_error, max_point_dist,
        min_cos_normal_error, options.check_num_images, &consistent_view_mask);
    if (num_consistent_views < options.min_num_views) {
        return false;
    }

    BitmapColor<uint8_t> color;
    view.bitmap->InterpolateNearestNeighbor(col / view.geom.scale_x,
                                            row / view.geom.scale_y, &color);

    const VoxelKey key = PointToVoxelKey(
        xyz, options.fusion_voxel_size, origin_x, origin_y, origin_z);
    voxels->operator[](key).Add(consistent_view_mask, xyz.x(), xyz.y(), xyz.z(),
                                normal.x(), normal.y(), normal.z(), color.r,
                                color.g, color.b);
    return true;
}

void FuseDenseConsistent(const std::vector<ViewData>& views,
                         const std::vector<std::vector<int>>& view_overlaps,
                         const DA3FusionOptions& options,
                         int thread_id,
                         int view_idx,
                         int row,
                         int col,
                         std::vector<std::vector<PlyPoint>>* task_points,
                         std::vector<std::vector<std::vector<int>>>* task_visibility) {
    const auto& view = views.at(view_idx);
    const float depth = view.depth_map->Get(row, col);
    if (depth <= 0.0f || !std::isfinite(depth)) {
        return;
    }

    const float max_squared_reproj_error =
        static_cast<float>(options.max_reproj_error * options.max_reproj_error);
    const float max_depth_error = static_cast<float>(options.max_depth_error);
    const float min_cos_normal_error =
        std::cos(DegToRad(static_cast<float>(options.max_normal_error)));

    const Eigen::Vector3f xyz =
        view.geom.inv_P *
        Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);
    const Eigen::Vector3f normal =
        view.geom.inv_R *
        Eigen::Vector3f(view.normal_map->Get(row, col, 0),
                        view.normal_map->Get(row, col, 1),
                        view.normal_map->Get(row, col, 2));

    int consistent_view_mask = 0;
    const std::vector<int>* overlapping_view_indices = nullptr;
    if (options.restrict_to_overlapping_views &&
        view_idx >= 0 &&
        view_idx < static_cast<int>(view_overlaps.size())) {
        overlapping_view_indices = &view_overlaps[view_idx];
    }
    const int num_consistent_views = CountConsistentViews(
        view, xyz, normal, views, overlapping_view_indices,
        max_squared_reproj_error, max_depth_error,
        static_cast<float>(options.max_point_dist), min_cos_normal_error,
        options.check_num_images, &consistent_view_mask);
    if (num_consistent_views < options.min_num_views) {
        return;
    }

    BitmapColor<uint8_t> color;
    view.bitmap->InterpolateNearestNeighbor(col / view.geom.scale_x,
                                            row / view.geom.scale_y, &color);

    PlyPoint point;
    point.x = xyz.x();
    point.y = xyz.y();
    point.z = xyz.z();
    point.nx = normal.x();
    point.ny = normal.y();
    point.nz = normal.z();
    point.r = color.r;
    point.g = color.g;
    point.b = color.b;

    (*task_points)[thread_id].push_back(point);
    (*task_visibility)[thread_id].push_back(
        ViewMaskToImageIndices(consistent_view_mask, views));
}

class DA3GraphFuser {
public:
    DA3GraphFuser(const std::vector<ViewData>& views,
                  const std::vector<std::vector<int>>& overlapping_images,
                  const DA3FusionOptions& options)
        : views_(views),
          overlapping_images_(overlapping_images),
          options_(options),
          max_squared_reproj_error_(static_cast<float>(
              options.max_reproj_error * options.max_reproj_error)),
          min_cos_normal_error_(std::cos(
              DegToRad(static_cast<float>(options.max_normal_error)))),
          max_depth_error_(static_cast<float>(options.max_depth_error)),
          max_point_dist_(static_cast<float>(options.max_point_dist)) {
        fused_pixel_masks_.resize(views.size());
        for (size_t i = 0; i < views.size(); ++i) {
            fused_pixel_masks_[i] =
                Mat<char>(views[i].geom.width, views[i].geom.height, 1);
        }
    }

    void Fuse(const int thread_id, const int view_idx, const int row,
              const int col) {
        auto& fused_pixel_mask = fused_pixel_masks_.at(view_idx);
        if (fused_pixel_mask.Get(row, col) > 0) {
            return;
        }

        std::vector<FusionData> fusion_queue;
        fusion_queue.emplace_back(view_idx, row, col, 0);

        Eigen::Vector4f fused_ref_point = Eigen::Vector4f::Zero();
        Eigen::Vector3f fused_ref_normal = Eigen::Vector3f::Zero();

        std::vector<float> fused_point_x;
        std::vector<float> fused_point_y;
        std::vector<float> fused_point_z;
        std::vector<float> fused_point_nx;
        std::vector<float> fused_point_ny;
        std::vector<float> fused_point_nz;
        std::vector<uint8_t> fused_point_r;
        std::vector<uint8_t> fused_point_g;
        std::vector<uint8_t> fused_point_b;
        std::unordered_set<int> fused_point_visibility;

        while (!fusion_queue.empty()) {
            const FusionData data = fusion_queue.back();
            fusion_queue.pop_back();

            const int image_idx = data.view_idx;
            const int cur_row = data.row;
            const int cur_col = data.col;
            const int traversal_depth = data.traversal_depth;

            if (fused_pixel_masks_.at(image_idx).Get(cur_row, cur_col) > 0) {
                continue;
            }

            const auto& view = views_.at(image_idx);
            const float depth = view.depth_map->Get(cur_row, cur_col);
            if (depth <= 0.0f || !std::isfinite(depth)) {
                continue;
            }

            if (traversal_depth > 0) {
                const Eigen::Vector3f proj =
                    view.geom.P * fused_ref_point;
                if (std::abs(proj.z()) <
                    std::numeric_limits<float>::epsilon()) {
                    continue;
                }

                const float sampled_depth = SampleDepthBilinear(
                    *view.depth_map, proj.x() / proj.z(), proj.y() / proj.z(),
                    view.geom.width, view.geom.height);
                if (sampled_depth <= 0.0f || !std::isfinite(sampled_depth)) {
                    continue;
                }

                const float depth_error =
                    std::abs((proj.z() - sampled_depth) / sampled_depth);
                if (depth_error >
                    static_cast<float>(options_.max_depth_error)) {
                    continue;
                }

                const float col_proj = proj.x() / proj.z();
                const float row_proj = proj.y() / proj.z();
                const float col_diff = col_proj - static_cast<float>(cur_col);
                const float row_diff = row_proj - static_cast<float>(cur_row);
                if (col_diff * col_diff + row_diff * row_diff >
                    max_squared_reproj_error_) {
                    continue;
                }
            }

            const Eigen::Vector3f normal_cam =
                Eigen::Vector3f(view.normal_map->Get(cur_row, cur_col, 0),
                                view.normal_map->Get(cur_row, cur_col, 1),
                                view.normal_map->Get(cur_row, cur_col, 2));
            const Eigen::Vector3f normal = view.geom.inv_R * normal_cam;

            if (traversal_depth > 0) {
                if (fused_ref_normal.dot(normal) < min_cos_normal_error_) {
                    continue;
                }
            }

            const Eigen::Vector3f xyz =
                view.geom.inv_P * Eigen::Vector4f(cur_col * depth,
                                                  cur_row * depth, depth, 1.0f);

            if (traversal_depth == 0) {
                int seed_view_mask = 0;
                const std::vector<int>* overlapping_view_indices = nullptr;
                if (options_.restrict_to_overlapping_views &&
                    image_idx >= 0 &&
                    image_idx < static_cast<int>(overlapping_images_.size())) {
                    overlapping_view_indices = &overlapping_images_[image_idx];
                }
                const int num_consistent = CountConsistentViews(
                    view, xyz, normal, views_, overlapping_view_indices,
                    max_squared_reproj_error_, max_depth_error_, max_point_dist_,
                    min_cos_normal_error_, options_.check_num_images,
                    &seed_view_mask);
                if (num_consistent < options_.min_num_views) {
                    return;
                }
            } else if ((xyz - fused_ref_point.head<3>()).norm() > max_point_dist_) {
                continue;
            }

            BitmapColor<uint8_t> color;
            view.bitmap->InterpolateNearestNeighbor(
                cur_col / view.geom.scale_x, cur_row / view.geom.scale_y,
                &color);

            fused_pixel_masks_.at(image_idx).Set(cur_row, cur_col, 1);

            fused_point_x.push_back(xyz.x());
            fused_point_y.push_back(xyz.y());
            fused_point_z.push_back(xyz.z());
            fused_point_nx.push_back(normal.x());
            fused_point_ny.push_back(normal.y());
            fused_point_nz.push_back(normal.z());
            fused_point_r.push_back(color.r);
            fused_point_g.push_back(color.g);
            fused_point_b.push_back(color.b);
            fused_point_visibility.insert(view.geom.image_idx);

            if (traversal_depth == 0) {
                fused_ref_point =
                    Eigen::Vector4f(xyz.x(), xyz.y(), xyz.z(), 1.0f);
                fused_ref_normal = normal;
            }

            if (fused_point_x.size() >=
                static_cast<size_t>(options_.max_num_pixels)) {
                break;
            }

            if (traversal_depth >= options_.max_traversal_depth - 1) {
                continue;
            }

            std::vector<int> neighbor_views;
            if (static_cast<size_t>(image_idx) < overlapping_images_.size() &&
                !overlapping_images_[image_idx].empty()) {
                neighbor_views = overlapping_images_[image_idx];
            } else {
                neighbor_views.reserve(views_.size());
                for (const auto& other : views_) {
                    if (other.geom.view_idx != image_idx) {
                        neighbor_views.push_back(other.geom.view_idx);
                    }
                }
            }

            int num_expanded = 0;
            for (const int next_view_idx : neighbor_views) {
                if (options_.check_num_images > 0 &&
                    num_expanded >= options_.check_num_images) {
                    break;
                }
                if (next_view_idx < 0 ||
                    next_view_idx >= static_cast<int>(views_.size())) {
                    continue;
                }

                const auto& next_view = views_.at(next_view_idx);
                const Eigen::Vector3f next_proj =
                    next_view.geom.P * xyz.homogeneous();
                if (std::abs(next_proj.z()) <
                    std::numeric_limits<float>::epsilon()) {
                    continue;
                }

                const int next_col = static_cast<int>(
                    std::lround(next_proj.x() / next_proj.z()));
                const int next_row = static_cast<int>(
                    std::lround(next_proj.y() / next_proj.z()));
                if (next_col < 0 || next_row < 0 ||
                    next_col >= next_view.geom.width ||
                    next_row >= next_view.geom.height) {
                    continue;
                }

                fusion_queue.emplace_back(next_view_idx, next_row, next_col,
                                          traversal_depth + 1);
                ++num_expanded;
            }
        }

        const size_t num_pixels = fused_point_x.size();
        if (num_pixels < static_cast<size_t>(options_.min_num_pixels)) {
            return;
        }

        std::unordered_set<int> view_mask;
        for (const int image_idx : fused_point_visibility) {
            for (const auto& view : views_) {
                if (view.geom.image_idx == image_idx) {
                    view_mask.insert(view.geom.view_idx);
                }
            }
        }
        if (static_cast<int>(view_mask.size()) < options_.min_num_views) {
            return;
        }

        PlyPoint fused_point;
        Eigen::Vector3f fused_normal;
        fused_normal.x() = Median(&fused_point_nx);
        fused_normal.y() = Median(&fused_point_ny);
        fused_normal.z() = Median(&fused_point_nz);
        const float fused_normal_norm = fused_normal.norm();
        if (fused_normal_norm < std::numeric_limits<float>::epsilon()) {
            return;
        }

        fused_point.x = Median(&fused_point_x);
        fused_point.y = Median(&fused_point_y);
        fused_point.z = Median(&fused_point_z);
        fused_point.nx = fused_normal.x() / fused_normal_norm;
        fused_point.ny = fused_normal.y() / fused_normal_norm;
        fused_point.nz = fused_normal.z() / fused_normal_norm;
        fused_point.r = static_cast<uint8_t>(
            std::lround(Median(&fused_point_r)));
        fused_point.g = static_cast<uint8_t>(
            std::lround(Median(&fused_point_g)));
        fused_point.b = static_cast<uint8_t>(
            std::lround(Median(&fused_point_b)));

        task_points_[thread_id].push_back(fused_point);
        task_visibility_[thread_id].emplace_back(fused_point_visibility.begin(),
                                                 fused_point_visibility.end());
    }

    void Collect(DA3FusionResult* result) const {
        size_t total = 0;
        for (const auto& pts : task_points_) {
            total += pts.size();
        }
        result->points.reserve(total);
        result->visibility.reserve(total);
        for (const auto& pts : task_points_) {
            result->points.insert(result->points.end(), pts.begin(), pts.end());
        }
        for (const auto& vis : task_visibility_) {
            result->visibility.insert(result->visibility.end(), vis.begin(),
                                      vis.end());
        }
    }

    void ResizeTasks(const int num_threads) {
        task_points_.resize(num_threads);
        task_visibility_.resize(num_threads);
    }

private:
    const std::vector<ViewData>& views_;
    const std::vector<std::vector<int>>& overlapping_images_;
    const DA3FusionOptions& options_;
    const float max_squared_reproj_error_;
    const float min_cos_normal_error_;
    const float max_depth_error_;
    const float max_point_dist_;
    std::vector<Mat<char>> fused_pixel_masks_;
    mutable std::vector<std::vector<PlyPoint>> task_points_;
    mutable std::vector<std::vector<std::vector<int>>> task_visibility_;
};

void OptionalVoxelMerge(DA3FusionResult* result, double voxel_size) {
    if (result == nullptr || voxel_size <= 0.0 || result->points.empty()) {
        return;
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    for (const auto& point : result->points) {
        min_x = std::min(min_x, static_cast<double>(point.x));
        min_y = std::min(min_y, static_cast<double>(point.y));
        min_z = std::min(min_z, static_cast<double>(point.z));
    }

    const double half = 0.5 * voxel_size;
    const double origin_x = min_x - half;
    const double origin_y = min_y - half;
    const double origin_z = min_z - half;

    struct Accum {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double nx = 0.0;
        double ny = 0.0;
        double nz = 0.0;
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        int count = 0;
        std::vector<int> visibility;

        void Add(const PlyPoint& point, const std::vector<int>& vis) {
            x += point.x;
            y += point.y;
            z += point.z;
            nx += point.nx;
            ny += point.ny;
            nz += point.nz;
            r += point.r;
            g += point.g;
            b += point.b;
            ++count;
            visibility.insert(visibility.end(), vis.begin(), vis.end());
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
            const double nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
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

        std::vector<int> UniqueVisibility() const {
            std::vector<int> out = visibility;
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        }
    };

    std::unordered_map<VoxelKey, Accum, VoxelKeyHash> voxels;
    voxels.reserve(result->points.size());
    for (size_t i = 0; i < result->points.size(); ++i) {
        const auto& point = result->points[i];
        const double ref_x = (static_cast<double>(point.x) - origin_x) / voxel_size;
        const double ref_y = (static_cast<double>(point.y) - origin_y) / voxel_size;
        const double ref_z = (static_cast<double>(point.z) - origin_z) / voxel_size;
        VoxelKey key{static_cast<int>(std::floor(ref_x)),
                     static_cast<int>(std::floor(ref_y)),
                     static_cast<int>(std::floor(ref_z))};
        voxels[key].Add(point, result->visibility[i]);
    }

    DA3FusionResult merged;
    merged.points.reserve(voxels.size());
    merged.visibility.reserve(voxels.size());
    for (const auto& [_, acc] : voxels) {
        merged.points.push_back(acc.Average());
        merged.visibility.push_back(acc.UniqueVisibility());
    }
    *result = std::move(merged);
}

void MergeVoxelMaps(
        const std::vector<std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash>>&
                thread_voxels,
        std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash>* merged) {
    for (const auto& local : thread_voxels) {
        for (const auto& [key, acc] : local) {
            auto& dst = (*merged)[key];
            dst.view_mask |= acc.view_mask;
            dst.count += acc.count;
            dst.x += acc.x;
            dst.y += acc.y;
            dst.z += acc.z;
            dst.nx += acc.nx;
            dst.ny += acc.ny;
            dst.nz += acc.nz;
            dst.r += acc.r;
            dst.g += acc.g;
            dst.b += acc.b;
        }
    }
}

void ComputeSceneOrigin(const std::vector<ViewData>& views,
                        int stride,
                        double voxel_size,
                        double* origin_x,
                        double* origin_y,
                        double* origin_z) {
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    const int bounds_stride = std::max(stride, 4);

    for (const auto& view : views) {
        for (int row = 0; row < view.geom.height; row += bounds_stride) {
            for (int col = 0; col < view.geom.width; col += bounds_stride) {
                const float depth = view.depth_map->Get(row, col);
                if (depth <= 0.0f || !std::isfinite(depth)) {
                    continue;
                }
                const Eigen::Vector3f xyz =
                    view.geom.inv_P *
                    Eigen::Vector4f(col * depth, row * depth, depth, 1.0f);
                min_x = std::min(min_x, static_cast<double>(xyz.x()));
                min_y = std::min(min_y, static_cast<double>(xyz.y()));
                min_z = std::min(min_z, static_cast<double>(xyz.z()));
            }
        }
    }

    if (!std::isfinite(min_x)) {
        min_x = min_y = min_z = 0.0;
    }
    const double half = 0.5 * voxel_size;
    *origin_x = min_x - half;
    *origin_y = min_y - half;
    *origin_z = min_z - half;
}

size_t CountValidDepthPixelsInViews(const std::vector<ViewData>& views,
                                    int stride) {
    size_t count = 0;
    for (const auto& view : views) {
        for (int row = 0; row < view.geom.height; row += stride) {
            for (int col = 0; col < view.geom.width; col += stride) {
                const float depth = view.depth_map->Get(row, col);
                if (depth > 0.0f && std::isfinite(depth)) {
                    ++count;
                }
            }
        }
    }
    return count;
}

}  // namespace

DA3FusionResult FuseDA3DepthMaps(const std::string& dense_workspace_path,
                                 const DA3FusionOptions& options,
                                 const std::string& input_type) {
    DA3FusionResult result;
    const int stride = std::max(1, options.pixel_stride);

    Workspace::Options workspace_options;
    workspace_options.workspace_path = dense_workspace_path;
    workspace_options.workspace_format = "COLMAP";
    workspace_options.input_type = input_type;
    workspace_options.image_as_rgb = true;
    workspace_options.max_image_size = options.max_image_size;
    workspace_options.num_threads = options.num_threads;
    Workspace workspace(workspace_options);

    const auto& model = workspace.GetModel();
    if (model.images.empty()) {
        LOG(ERROR) << "DA3 fusion: no images in workspace model";
        return result;
    }

    const auto image_names = ReadTextFileLines(
        JoinPaths(dense_workspace_path, workspace_options.stereo_folder,
                  "fusion.cfg"));
    if (image_names.empty()) {
        LOG(ERROR) << "DA3 fusion: fusion.cfg is empty";
        return result;
    }
    workspace.Load(image_names);

    std::vector<ViewData> views;
    views.reserve(image_names.size());
    for (const auto& image_name : image_names) {
        const int image_idx = model.GetImageIdx(image_name);
        if (!workspace.HasDepthMap(image_idx) ||
            !workspace.HasNormalMap(image_idx) ||
            !workspace.HasBitmap(image_idx)) {
            continue;
        }

        const auto& image = model.images.at(static_cast<size_t>(image_idx));
        const auto& depth_map = workspace.GetDepthMap(image_idx);
        const int width = static_cast<int>(depth_map.GetWidth());
        const int height = static_cast<int>(depth_map.GetHeight());
        if (width <= 0 || height <= 0) {
            continue;
        }

        ViewData view;
        view.geom.view_idx = static_cast<int>(views.size());
        view.geom.image_idx = image_idx;
        view.geom.width = width;
        view.geom.height = height;
        view.geom.scale_x =
            static_cast<float>(width) / static_cast<float>(image.GetWidth());
        view.geom.scale_y =
            static_cast<float>(height) / static_cast<float>(image.GetHeight());
        view.depth_map = &depth_map;
        view.normal_map = &workspace.GetNormalMap(image_idx);
        view.bitmap = &workspace.GetBitmap(image_idx);

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> K =
                Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
                        image.GetK());
        K(0, 0) *= view.geom.scale_x;
        K(0, 2) *= view.geom.scale_x;
        K(1, 1) *= view.geom.scale_y;
        K(1, 2) *= view.geom.scale_y;

        ComposeProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                                view.geom.P.data());
        ComposeInverseProjectionMatrix(K.data(), image.GetR(), image.GetT(),
                                       view.geom.inv_P.data());
        view.geom.inv_R =
                Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(
                        image.GetR())
                        .transpose();
        views.push_back(view);
    }

    if (views.empty()) {
        LOG(ERROR) << "DA3 fusion: no valid depth/normal maps loaded";
        return result;
    }

    std::vector<std::vector<int>> overlapping_images = model.GetMaxOverlappingImages(
        static_cast<size_t>(std::max(1, options.check_num_images)), 0.0);
    if (overlapping_images.size() < views.size()) {
        overlapping_images.resize(views.size());
    }

    std::vector<std::vector<int>> view_overlaps(views.size());
    for (auto& neighbors : view_overlaps) {
        neighbors.reserve(views.size());
    }
    for (size_t image_idx = 0; image_idx < overlapping_images.size();
         ++image_idx) {
        int view_idx = -1;
        for (const auto& view : views) {
            if (view.geom.image_idx == static_cast<int>(image_idx)) {
                view_idx = view.geom.view_idx;
                break;
            }
        }
        if (view_idx < 0) {
            continue;
        }
        for (const int neighbor_image_idx : overlapping_images[image_idx]) {
            for (const auto& view : views) {
                if (view.geom.image_idx == neighbor_image_idx) {
                    view_overlaps[view_idx].push_back(view.geom.view_idx);
                }
            }
        }
        if (view_overlaps[view_idx].empty()) {
            for (const auto& view : views) {
                if (view.geom.view_idx != view_idx) {
                    view_overlaps[view_idx].push_back(view.geom.view_idx);
                }
            }
        }
    }

    const int num_threads = GetEffectiveNumThreads(options.num_threads);
    ThreadPool thread_pool(num_threads);
    const int kRowStride = 10;

    auto RunVoxelConsensus = [&](DA3FusionResult* out) {
        if (options.fusion_voxel_size <= 0.0) {
            LOG(ERROR) << "DA3 fusion: voxel consensus requires fusion_voxel_size > 0";
            return;
        }

        double origin_x = 0.0;
        double origin_y = 0.0;
        double origin_z = 0.0;
        ComputeSceneOrigin(views, stride, options.fusion_voxel_size, &origin_x,
                           &origin_y, &origin_z);

        std::vector<std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash>>
            thread_voxels(num_threads);
        std::vector<size_t> thread_accepted(num_threads, 0);
        std::vector<size_t> thread_skipped(num_threads, 0);

        auto ProcessVoxelRows = [&](const int row_start, const int height,
                                    const int width, const int view_idx) {
            const int row_end = std::min(height, row_start + kRowStride);
            for (int row = row_start; row < row_end; row += stride) {
                for (int col = 0; col < width; col += stride) {
                    const int thread_id = thread_pool.GetThreadIndex();
                    const auto& view = views.at(view_idx);
                    const float depth = view.depth_map->Get(row, col);
                    if (depth <= 0.0f || !std::isfinite(depth)) {
                        continue;
                    }
                    if (AccumulateVoxelSample(views, view_overlaps, options,
                                              origin_x, origin_y, origin_z,
                                              view_idx, row, col,
                                              &thread_voxels[thread_id])) {
                        ++thread_accepted[thread_id];
                    } else {
                        ++thread_skipped[thread_id];
                    }
                }
            }
        };

        RECON_LOG_DEBUG(
                "DA3 fusion: voxel consensus (voxel=%fm, stride=%d, "
                "min_views=%d, reproj=%fpx, depth_err=%f, point_dist=%fm, "
                "normal_err=%fdeg, threads=%d)\n",
                options.fusion_voxel_size, stride, options.min_num_views,
                options.max_reproj_error, options.max_depth_error,
                options.max_point_dist, options.max_normal_error, num_threads);

        for (const auto& view : views) {
            for (int row_start = 0; row_start < view.geom.height;
                 row_start += kRowStride) {
                thread_pool.AddTask(ProcessVoxelRows, row_start, view.geom.height,
                                    view.geom.width, view.geom.view_idx);
            }
        }
        thread_pool.Wait();

        size_t accepted = 0;
        size_t skipped = 0;
        for (int t = 0; t < num_threads; ++t) {
            accepted += thread_accepted[t];
            skipped += thread_skipped[t];
        }

        std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> merged_voxels;
        merged_voxels.reserve(accepted / 4 + 1);
        MergeVoxelMaps(thread_voxels, &merged_voxels);

        out->points.clear();
        out->visibility.clear();
        out->points.reserve(merged_voxels.size());
        out->visibility.reserve(merged_voxels.size());
        size_t emitted = 0;
        for (const auto& [_, acc] : merged_voxels) {
            if (acc.NumViews() < options.min_num_views) {
                continue;
            }
            out->points.push_back(acc.Average());
            out->visibility.push_back(acc.VisibleImageIndices(views));
            ++emitted;
        }

        out->num_accepted_samples = accepted;
        out->num_skipped_samples = skipped;

        RECON_LOG_DEBUG(
                "DA3 fusion: voxel accepted %zu samples (skipped %zu "
                "inconsistent) -> %zu voxel points\n",
                accepted, skipped, emitted);
    };

    auto RunDenseFusion = [&](DA3FusionResult* out) {
        std::vector<std::vector<PlyPoint>> task_points(num_threads);
        std::vector<std::vector<std::vector<int>>> task_visibility(num_threads);
        std::vector<size_t> thread_accepted(num_threads, 0);
        std::vector<size_t> thread_skipped(num_threads, 0);

        auto ProcessDenseRows = [&](const int row_start, const int height,
                                  const int width, const int view_idx) {
            const int row_end = std::min(height, row_start + kRowStride);
            for (int row = row_start; row < row_end; row += stride) {
                for (int col = 0; col < width; col += stride) {
                    const int thread_id = thread_pool.GetThreadIndex();
                    const auto& view = views.at(view_idx);
                    const float depth = view.depth_map->Get(row, col);
                    if (depth <= 0.0f || !std::isfinite(depth)) {
                        continue;
                    }
                    const size_t before =
                        task_points[thread_id].size();
                    FuseDenseConsistent(views, view_overlaps, options, thread_id,
                                        view_idx, row, col, &task_points,
                                        &task_visibility);
                    if (task_points[thread_id].size() > before) {
                        ++thread_accepted[thread_id];
                    } else {
                        ++thread_skipped[thread_id];
                    }
                }
            }
        };

        RECON_LOG_DEBUG(
                "DA3 fusion: dense multi-view (stride=%d, min_views=%d, "
                "reproj=%fpx, depth_err=%f, point_dist=%fm, normal_err=%fdeg, "
                "threads=%d)\n",
                stride, options.min_num_views, options.max_reproj_error,
                options.max_depth_error, options.max_point_dist,
                options.max_normal_error, num_threads);

        for (const auto& view : views) {
            for (int row_start = 0; row_start < view.geom.height;
                 row_start += kRowStride) {
                thread_pool.AddTask(ProcessDenseRows, row_start, view.geom.height,
                                    view.geom.width, view.geom.view_idx);
            }
        }
        thread_pool.Wait();

        out->points.clear();
        out->visibility.clear();
        for (int t = 0; t < num_threads; ++t) {
            out->points.insert(out->points.end(), task_points[t].begin(),
                               task_points[t].end());
            out->visibility.insert(out->visibility.end(),
                                   task_visibility[t].begin(),
                                   task_visibility[t].end());
        }
        size_t accepted = 0;
        size_t skipped = 0;
        for (int t = 0; t < num_threads; ++t) {
            accepted += thread_accepted[t];
            skipped += thread_skipped[t];
        }
        RECON_LOG_DEBUG(
                "DA3 fusion: dense accepted %zu samples (skipped %zu "
                "inconsistent)\n",
                accepted, skipped);
    };

    auto RunGraphFusion = [&](DA3FusionResult* out) {
        DA3GraphFuser fuser(views, view_overlaps, options);
        fuser.ResizeTasks(num_threads);

        RECON_LOG_DEBUG(
                "DA3 fusion: graph traversal (stride=%d, min_pixels=%d, "
                "min_views=%d, reproj=%fpx, depth_err=%f, normal_err=%fdeg, "
                "threads=%d)\n",
                stride, options.min_num_pixels, options.min_num_views,
                options.max_reproj_error, options.max_depth_error,
                options.max_normal_error, num_threads);

        auto ProcessViewRows = [&](DA3GraphFuser& active_fuser,
                                   const int row_start, const int height,
                                   const int width, const int view_idx) {
            const int row_end = std::min(height, row_start + kRowStride);
            for (int row = row_start; row < row_end; row += stride) {
                for (int col = 0; col < width; col += stride) {
                    const int thread_id = thread_pool.GetThreadIndex();
                    active_fuser.Fuse(thread_id, view_idx, row, col);
                }
            }
        };

        for (const auto& view : views) {
            for (int row_start = 0; row_start < view.geom.height;
                 row_start += kRowStride) {
                thread_pool.AddTask(ProcessViewRows, std::ref(fuser), row_start,
                                    view.geom.height, view.geom.width,
                                    view.geom.view_idx);
            }
        }
        thread_pool.Wait();
        fuser.Collect(out);
    };

    result.num_valid_depth_pixels =
        CountValidDepthPixelsInViews(views, stride);

    if (options.use_voxel_consensus) {
        RunVoxelConsensus(&result);
    } else if (options.use_dense_fusion) {
        RunDenseFusion(&result);
    } else {
        RunGraphFusion(&result);
    }

    const size_t before_voxel = result.points.size();
    if (!options.use_voxel_consensus) {
        OptionalVoxelMerge(&result, options.fusion_voxel_size);
    }

    if (!options.use_voxel_consensus && options.fusion_voxel_size > 0.0) {
        RECON_LOG_DEBUG(
                "DA3 fusion: %zu fused points -> %zu after optional voxel "
                "(%fm)\n",
                before_voxel, result.points.size(),
                options.fusion_voxel_size);
    } else {
        RECON_LOG_DEBUG("DA3 fusion: %zu fused points\n", before_voxel);
    }
    return result;
}

}  // namespace mvs
}  // namespace colmap
