// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Nearest-splat color lookup over a voxel hash grid of the gaussian splat
// centers. Shared by the Sam3dWorker (PBR assembly for the GLB bake) and the
// GUI thread (DB vertex-color fallback mesh).

#include <QVector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sam3d_colors {

class SplatColorGrid {
public:
    SplatColorGrid(const QVector<float>& centers, const QVector<float>& rgb)
        : m_centers(centers.constData()), m_rgb(rgb.constData()) {
        const int64_t n = centers.size() / 3;
        m_buckets.reserve(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            m_buckets[cellKey(centers[i * 3], centers[i * 3 + 1],
                              centers[i * 3 + 2])]
                    .push_back(static_cast<int32_t>(i));
        }
        // Average color: fallback for the (rare) vertices whose neighborhood
        // is splat-free even at the widest search radius.
        for (int c = 0; c < 3; ++c) {
            double sum = 0.0;
            for (int64_t i = 0; i < n; ++i) sum += m_rgb[i * 3 + c];
            m_avg[c] = static_cast<float>(sum / static_cast<double>(n));
        }
    }

    static constexpr float kCell = 0.02f;

    bool colorAt(float x, float y, float z, float out[3]) const {
        const auto cx = cellOf(x);
        const auto cy = cellOf(y);
        const auto cz = cellOf(z);
        std::vector<int32_t> candidates;
        for (const int r : {0, 1, 2, 3, 5, 8}) {
            for (int dz = -r; dz <= r; ++dz) {
                for (int dy = -r; dy <= r; ++dy) {
                    for (int dx = -r; dx <= r; ++dx) {
                        // Shell iteration: visit each cell exactly once.
                        if (std::max({std::abs(dx), std::abs(dy),
                                      std::abs(dz)}) != r) {
                            continue;
                        }
                        const auto it = m_buckets.find(
                                cellKeyAt(cx + dx, cy + dy, cz + dz));
                        if (it != m_buckets.end()) {
                            candidates.insert(candidates.end(),
                                              it->second.begin(),
                                              it->second.end());
                        }
                    }
                }
            }
            if (!candidates.empty()) {
                float best = std::numeric_limits<float>::infinity();
                int32_t bestIdx = -1;
                for (const int32_t idx : candidates) {
                    const float* c = m_centers + static_cast<size_t>(idx) * 3;
                    const float d2 = (c[0] - x) * (c[0] - x) +
                                     (c[1] - y) * (c[1] - y) +
                                     (c[2] - z) * (c[2] - z);
                    if (d2 < best) {
                        best = d2;
                        bestIdx = idx;
                    }
                }
                for (int c = 0; c < 3; ++c) {
                    out[c] = m_rgb[static_cast<size_t>(bestIdx) * 3 + c];
                }
                return true;
            }
        }
        out[0] = m_avg[0];
        out[1] = m_avg[1];
        out[2] = m_avg[2];
        return false;
    }

private:
    static int32_t cellOf(float v) {
        return static_cast<int32_t>(std::floor(v / kCell));
    }
    // 21 bits per signed axis offset; the unit-scale object domain keeps the
    // cell coordinates far inside that range.
    static uint64_t cellKeyAt(int32_t x, int32_t y, int32_t z) {
        const uint64_t ux = static_cast<uint64_t>(
                static_cast<uint32_t>(x + (1 << 20)) & 0x1FFFFF);
        const uint64_t uy = static_cast<uint64_t>(
                static_cast<uint32_t>(y + (1 << 20)) & 0x1FFFFF);
        const uint64_t uz = static_cast<uint64_t>(
                static_cast<uint32_t>(z + (1 << 20)) & 0x1FFFFF);
        return (ux << 42) | (uy << 21) | uz;
    }
    static uint64_t cellKey(float x, float y, float z) {
        return cellKeyAt(cellOf(x), cellOf(y), cellOf(z));
    }

    const float* m_centers;
    const float* m_rgb;
    std::unordered_map<uint64_t, std::vector<int32_t>> m_buckets;
    float m_avg[3] = {0.5f, 0.5f, 0.5f};
};

}  // namespace sam3d_colors
