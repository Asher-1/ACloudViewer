#include "voxelization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace openpcdet {

namespace {

// 64-bit key from grid coord (z,y,x). grid_size is small (typically
// [432,496,1] for kitti), so 21 bits per axis is plenty.
inline uint64_t coord_key(int x, int y, int z) {
    return (uint64_t) (uint32_t) z << 42 |
           (uint64_t) (uint32_t) y << 21 |
           (uint64_t) (uint32_t) x;
}

// Flat open-addressing coordinate table (same design as the sparse-rulebook
// coord_table): key and value packed in one 16-byte slot, load factor <= 0.5,
// plain low-bit hash. A probe is a single cache-line compare, which is ~5x
// faster than std::unordered_map's node lookups and dominates voxelization
// wall time on dense sweeps (34k points -> ~10 ms of map probes).
//
// Unlike the rulebook table, voxel keys are packed grid coordinates
// (z<<42|y<<21|x) that are sparse in the low bits, so the raw low-bit mask
// would collapse every distinct y onto the same bucket. A multiply-mix
// (Knuth multiplicative hashing) scrambles all 64 bits before masking.
class voxel_table {
public:
    struct slot {
        uint64_t key;
        int32_t value;
        int32_t padding;
    };

    void build(int64_t count) {
        int64_t capacity = 1;
        while (capacity < std::max<int64_t>(8, count * 2)) capacity <<= 1;
        slots_.assign(static_cast<size_t>(capacity), slot{0, -1});
        mask_ = static_cast<uint32_t>(capacity) - 1;
    }

    // Returns the stored slot, or inserts `value` and returns it. Empty
    // slots are marked by value == -1 (the packed key of voxel (0,0,0) is 0).
    int32_t find_or_insert(uint64_t key, int32_t value) {
        size_t idx = hash_slot(key);
        for (;;) {
            if (slots_[idx].value < 0) {
                slots_[idx] = slot{key, value};
                return value;
            }
            if (slots_[idx].key == key) return slots_[idx].value;
            idx = (idx + 1) & mask_;
        }
    }

    // Lookup only; -1 when absent (used once the voxel grid is full, so the
    // table never grows past its reserved capacity).
    int32_t find(uint64_t key) const {
        size_t idx = hash_slot(key);
        for (;;) {
            if (slots_[idx].value < 0) return -1;
            if (slots_[idx].key == key) return slots_[idx].value;
            idx = (idx + 1) & mask_;
        }
    }

private:
    std::vector<slot> slots_;
    uint32_t mask_ = 0;

    size_t hash_slot(uint64_t key) const {
        const uint64_t mixed = key * 0x9E3779B97F4A7C15ULL;
        // Knuth multiplicative hashing: the low bits of the product only
        // depend on the low input bits (grid coords are packed into the high
        // bits), so bucket from the mixed high bits instead.
        return static_cast<size_t>(mixed >> 32) & mask_;
    }
};

} // namespace

bool voxelize(const voxelization_params & p,
              const float * points, int32_t n_points, int32_t point_dim,
              voxel_output & out, std::string & err) {
    if (p.point_cloud_range.size() != 6) { err = "bad point_cloud_range"; return false; }
    if (p.voxel_size.size() != 3)        { err = "bad voxel_size"; return false; }
    if (p.grid_size.size() != 3)         { err = "bad grid_size"; return false; }
    if (n_points < 0 || point_dim < 3 || (n_points > 0 && !points)) {
        err = "bad point buffer"; return false;
    }
    if (p.max_points_per_voxel <= 0)     { err = "bad max_points_per_voxel"; return false; }
    if (p.max_voxels <= 0)               { err = "bad max_voxels"; return false; }

    const float x0 = p.point_cloud_range[0], y0 = p.point_cloud_range[1], z0 = p.point_cloud_range[2];
    const float x1 = p.point_cloud_range[3], y1 = p.point_cloud_range[4], z1 = p.point_cloud_range[5];
    const float vx = p.voxel_size[0], vy = p.voxel_size[1], vz = p.voxel_size[2];
    const int   nx = p.grid_size[0], ny = p.grid_size[1], nz = p.grid_size[2];
    const int   mpv = p.max_points_per_voxel;
    const int   min_pts = std::max(1, p.min_points_per_voxel);

    out.voxels.assign((size_t) p.max_voxels * mpv * point_dim, 0.0f);
    out.num_points.assign((size_t) p.max_voxels, 0);
    out.coords.assign((size_t) p.max_voxels * 4, 0);
    out.actual_voxels = 0;

    // One entry per voxel: index into out.voxels.
    voxel_table voxel_idx;
    voxel_idx.build(p.max_voxels);

    for (int32_t i = 0; i < n_points; i++) {
        const float * pt = points + (size_t) i * point_dim;
        const float px = pt[0], py = pt[1], pz = pt[2];
        if (px < x0 || px >= x1 || py < y0 || py >= y1 || pz < z0 || pz >= z1) continue;

        const int gx = (int) std::floor((px - x0) / vx);
        const int gy = (int) std::floor((py - y0) / vy);
        const int gz = (int) std::floor((pz - z0) / vz);
        if (gx < 0 || gx >= nx || gy < 0 || gy >= ny || gz < 0 || gz >= nz) continue;

        const uint64_t key = coord_key(gx, gy, gz);
        int32_t slot;
        if (out.actual_voxels >= p.max_voxels) {
            // Voxel grid full: look up without growing the table.
            const int32_t found = voxel_idx.find(key);
            if (found < 0) continue;  // new voxel while grid is full -> drop
            slot = found;
        } else {
            slot = voxel_idx.find_or_insert(key, out.actual_voxels);
            if (slot == out.actual_voxels) {
                // coords: [batch_idx=0, z, y, x]  — matches OpenPCDet's voxel_coords layout
                out.coords[(size_t) slot * 4 + 0] = 0;
                out.coords[(size_t) slot * 4 + 1] = gz;
                out.coords[(size_t) slot * 4 + 2] = gy;
                out.coords[(size_t) slot * 4 + 3] = gx;
                ++out.actual_voxels;
            }
        }

        int & np = out.num_points[slot];
        if (np >= mpv) continue;  // voxel full, drop this point (matches VoxelGenerator)
        float * dst = out.voxels.data() + ((size_t) slot * mpv + np) * point_dim;
        for (int d = 0; d < point_dim; d++) dst[d] = pt[d];
        np++;
    }

    // Drop voxels with too few points (compaction in place).
    int32_t kept = 0;
    for (int32_t s = 0; s < out.actual_voxels; s++) {
        if (out.num_points[s] < min_pts) continue;
        if (kept != s) {
            std::copy(out.voxels.begin() + (size_t) s * mpv * point_dim,
                      out.voxels.begin() + (size_t) (s + 1) * mpv * point_dim,
                      out.voxels.begin() + (size_t) kept * mpv * point_dim);
            out.num_points[kept] = out.num_points[s];
            for (int d = 0; d < 4; d++)
                out.coords[(size_t) kept * 4 + d] = out.coords[(size_t) s * 4 + d];
        }
        kept++;
    }
    out.actual_voxels = kept;
    return true;
}

bool voxelize_mean(const voxelization_params & p,
                   const float * points, int32_t n_points, int32_t point_dim,
                   mean_voxel_output & out, std::string & err) {
    if (p.point_cloud_range.size() != 6) { err = "bad point_cloud_range"; return false; }
    if (p.voxel_size.size() != 3)        { err = "bad voxel_size"; return false; }
    if (p.grid_size.size() != 3)         { err = "bad grid_size"; return false; }
    if (n_points < 0 || point_dim < 3 || (n_points > 0 && !points)) {
        err = "bad point buffer"; return false;
    }
    if (p.max_points_per_voxel <= 0)     { err = "bad max_points_per_voxel"; return false; }
    if (p.max_voxels <= 0)               { err = "bad max_voxels"; return false; }

    const float x0 = p.point_cloud_range[0], y0 = p.point_cloud_range[1], z0 = p.point_cloud_range[2];
    const float x1 = p.point_cloud_range[3], y1 = p.point_cloud_range[4], z1 = p.point_cloud_range[5];
    const float vx = p.voxel_size[0], vy = p.voxel_size[1], vz = p.voxel_size[2];
    const int nx = p.grid_size[0], ny = p.grid_size[1], nz = p.grid_size[2];
    const int min_pts = std::max(1, p.min_points_per_voxel);

    out.features.assign(static_cast<size_t>(p.max_voxels) * point_dim, 0.0f);
    std::vector<int32_t> num_points(static_cast<size_t>(p.max_voxels), 0);
    out.coords.assign(static_cast<size_t>(p.max_voxels) * 4, 0);
    out.actual_voxels = 0;

    voxel_table voxel_idx;
    voxel_idx.build(p.max_voxels);
    for (int32_t i = 0; i < n_points; ++i) {
        const float * point = points + static_cast<size_t>(i) * point_dim;
        const float px = point[0], py = point[1], pz = point[2];
        if (px < x0 || px >= x1 || py < y0 || py >= y1 || pz < z0 || pz >= z1) continue;

        const int gx = static_cast<int>(std::floor((px - x0) / vx));
        const int gy = static_cast<int>(std::floor((py - y0) / vy));
        const int gz = static_cast<int>(std::floor((pz - z0) / vz));
        if (gx < 0 || gx >= nx || gy < 0 || gy >= ny || gz < 0 || gz >= nz) continue;

        const uint64_t key = coord_key(gx, gy, gz);
        int32_t slot;
        if (out.actual_voxels >= p.max_voxels) {
            slot = voxel_idx.find(key);
            if (slot < 0) continue;
        } else {
            slot = voxel_idx.find_or_insert(key, out.actual_voxels);
            if (slot == out.actual_voxels) {
                int32_t * coord = out.coords.data() + static_cast<size_t>(slot) * 4;
                coord[0] = 0; coord[1] = gz; coord[2] = gy; coord[3] = gx;
                ++out.actual_voxels;
            }
        }

        int32_t & count = num_points[slot];
        if (count >= p.max_points_per_voxel) continue;
        float * feature = out.features.data() + static_cast<size_t>(slot) * point_dim;
        for (int channel = 0; channel < point_dim; ++channel) feature[channel] += point[channel];
        ++count;
    }

    int32_t kept = 0;
    for (int32_t slot = 0; slot < out.actual_voxels; ++slot) {
        const int32_t count = num_points[slot];
        if (count < min_pts) continue;
        float * source = out.features.data() + static_cast<size_t>(slot) * point_dim;
        if (kept != slot) {
            std::copy_n(source, point_dim,
                        out.features.data() + static_cast<size_t>(kept) * point_dim);
            std::copy_n(out.coords.data() + static_cast<size_t>(slot) * 4, 4,
                        out.coords.data() + static_cast<size_t>(kept) * 4);
        }
        float * feature = out.features.data() + static_cast<size_t>(kept) * point_dim;
        for (int channel = 0; channel < point_dim; ++channel) feature[channel] /= count;
        ++kept;
    }
    out.actual_voxels = kept;
    out.features.resize(static_cast<size_t>(kept) * point_dim);
    out.coords.resize(static_cast<size_t>(kept) * 4);
    return true;
}

} // namespace openpcdet
