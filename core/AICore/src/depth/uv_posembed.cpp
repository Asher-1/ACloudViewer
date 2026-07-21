// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "uv_posembed.hpp"

#include <cmath>
#include <vector>

namespace aicore {
namespace depth {

std::vector<float> uv_pos_embed(
        int pw, int ph, int C, float aspect, float omega0) {
    std::vector<float> out((size_t)ph * pw * C, 0.f);
    if (pw <= 0 || ph <= 0 || C <= 0) return out;

    // create_uv_grid spans
    const double diag = std::sqrt((double)aspect * aspect + 1.0);
    const double span_x = (double)aspect / diag;
    const double span_y = 1.0 / diag;
    const double left_x = -span_x * (pw - 1) / (double)pw;
    const double right_x = span_x * (pw - 1) / (double)pw;
    const double top_y = -span_y * (ph - 1) / (double)ph;
    const double bottom_y = span_y * (ph - 1) / (double)ph;

    // linspace coordinates
    std::vector<double> x_coords(pw), y_coords(ph);
    if (pw == 1) {
        x_coords[0] = left_x;
    } else {
        const double step = (right_x - left_x) / (pw - 1);
        for (int i = 0; i < pw; ++i) x_coords[i] = left_x + i * step;
    }
    if (ph == 1) {
        y_coords[0] = top_y;
    } else {
        const double step = (bottom_y - top_y) / (ph - 1);
        for (int i = 0; i < ph; ++i) y_coords[i] = top_y + i * step;
    }

    // Per-axis embedding: D = C/2 channels per axis (must be even), F = D/2
    // freqs.
    const int D = C / 2;  // 32 for C=64
    const int F = D / 2;  // 16 for C=64
    std::vector<double> omega(F);
    for (int j = 0; j < F; ++j) {
        // arange(D//2)/(D/2.0) -> j / (D/2.0); D/2.0 == F so exponent = j/F
        const double e = (double)j / ((double)D / 2.0);
        omega[j] = 1.0 / std::pow((double)omega0, e);
    }

    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            const double xc = x_coords[x];
            const double yc = y_coords[y];
            const size_t base = ((size_t)y * pw + x) * C;
            // emb_x: [sin(F), cos(F)] from x-coord -> channels [0, D)
            for (int j = 0; j < F; ++j) {
                const double o = xc * omega[j];
                out[base + j] = (float)std::sin(o);
                out[base + F + j] = (float)std::cos(o);
            }
            // emb_y: [sin(F), cos(F)] from y-coord -> channels [D, 2D)
            for (int j = 0; j < F; ++j) {
                const double o = yc * omega[j];
                out[base + D + j] = (float)std::sin(o);
                out[base + D + F + j] = (float)std::cos(o);
            }
        }
    }
    return out;
}

}  // namespace depth
}  // namespace aicore
