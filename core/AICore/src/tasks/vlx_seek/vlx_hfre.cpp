// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "vlx_hfre.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

namespace vlx {

// Line-for-line port of torchvision 0.25 cpu/roi_align_common.h
// pre_calc_for_bilinear_interpolate. Edge handling is the part that matters:
// a sample with y_low >= H-1 snaps to row H-1 with full weight (it is NOT
// zero-padded), so boxes touching the bottom/right border keep their mass.
void pre_calc_bilinear(int height, int width, int pooled, float roi_start_h,
                       float roi_start_w, float bin_h, float bin_w, int grid_h,
                       int grid_w, std::vector<PreCalc>* out) {
    out->clear();
    out->reserve(size_t(pooled) * pooled * grid_h * grid_w);
    for (int ph = 0; ph < pooled; ph++)
        for (int pw = 0; pw < pooled; pw++)
            for (int iy = 0; iy < grid_h; iy++) {
                const float yy = roi_start_h + ph * bin_h +
                                 float(iy + .5f) * bin_h / float(grid_h);
                for (int ix = 0; ix < grid_w; ix++) {
                    const float xx = roi_start_w + pw * bin_w +
                                     float(ix + .5f) * bin_w / float(grid_w);
                    float x = xx, y = yy;
                    if (y < -1.0 || y > height || x < -1.0 || x > width) {
                        out->push_back({0, 0, 0, 0, 0.f, 0.f, 0.f, 0.f});
                        continue;
                    }
                    if (y <= 0) y = 0;
                    if (x <= 0) x = 0;
                    int y_low = int(y), x_low = int(x), y_high, x_high;
                    if (y_low >= height - 1) { y_high = y_low = height - 1; y = float(y_low); }
                    else y_high = y_low + 1;
                    if (x_low >= width - 1) { x_high = x_low = width - 1; x = float(x_low); }
                    else x_high = x_low + 1;
                    const float ly = y - y_low, lx = x - x_low;
                    const float hy = 1. - ly, hx = 1. - lx;
                    out->push_back({y_low * width + x_low, y_low * width + x_high,
                                    y_high * width + x_low, y_high * width + x_high,
                                    hy * hx, hy * lx, ly * hx, ly * lx});
                }
            }
}

HfreEncoder::HfreEncoder(const HfreConfig& cfg) : cfg_(cfg) {}

void HfreEncoder::roi_align_mean_pool(const FeatureLevel& level,
                                      const std::vector<Box>& boxes_px,
                                      float spatial_scale, int out_size,
                                      bool aligned,
                                      std::vector<float>* out_nxc,
                                      int n_threads) {
    const int C = level.channels, H = level.height, W = level.width;
    const size_t N = boxes_px.size();
    out_nxc->assign(N * C, 0.f);
    // roi_align(...).mean(dim=(2,3)) == torchvision forward kernel (per-bin
    // sum of 4-tap products, /count) followed by ATen's CPU mean (sum, then
    // div by out_size^2). Cost: N * C * out^2 * grid^2 * 4 MACs; boxes are
    // independent, so they are split across threads.
    auto run_box = [&](size_t bi, std::vector<PreCalc>* pc) {
        const Box& b = boxes_px[bi];
        const float offset = aligned ? 0.5f : 0.f;
        const float rsw = b.x1 * spatial_scale - offset;
        const float rsh = b.y1 * spatial_scale - offset;
        float rw = (b.x2 * spatial_scale - offset) - rsw;
        float rh = (b.y2 * spatial_scale - offset) - rsh;
        if (!aligned) { rw = std::max(rw, 1.f); rh = std::max(rh, 1.f); }
        const float bin_h = rh / float(out_size), bin_w = rw / float(out_size);
        // Degenerate (negative-extent) boxes reach here when aligned=true
        // skips the max(rw,1) guard: clamp the grid to zero so the bin
        // loops stay empty and the reserve stays sane — matches the old
        // behavior (per_bin <= 0 contributed nothing) without the size_t
        // underflow the raw negative product caused in reserve().
        const int grid_h = std::max(0, int(std::ceil(rh / out_size)));
        const int grid_w = std::max(0, int(std::ceil(rw / out_size)));
        const float count = float(std::max(grid_h * grid_w, 1));
        pre_calc_bilinear(H, W, out_size, rsh, rsw, bin_h, bin_w, grid_h, grid_w, pc);
        const int per_bin = grid_h * grid_w;
        const float denom = float(out_size * out_size);
        for (int c = 0; c < C; ++c) {
            const float* in = &level.data[size_t(c) * H * W];
            // ATen's CPU kernel accumulates in float (cascaded per-bin sums,
            // each divided by count, then one global /size^2) — mirror that
            // instead of the previous exact double pass: it is the reference
            // behavior AND vectorizes (~2x on the 3.6 GMAC hot loop).
            float sum = 0.f;
            size_t k = 0;
            for (int bin = 0; bin < out_size * out_size; ++bin) {
                float v = 0.f;
                for (int s = 0; s < per_bin; ++s, ++k) {
                    const PreCalc& p = (*pc)[k];
                    v += p.w1 * in[p.pos1] + p.w2 * in[p.pos2] +
                         p.w3 * in[p.pos3] + p.w4 * in[p.pos4];
                }
                sum += v / count;
            }
            (*out_nxc)[bi * C + c] = sum / denom;
        }
    };
    const int T = std::max(1, std::min<int>(n_threads, int(N)));
    std::vector<std::thread> pool;
    for (int t = 0; t < T; ++t)
        pool.emplace_back([&, t] {
            std::vector<PreCalc> pc;
            for (size_t bi = t; bi < N; bi += T) run_box(bi, &pc);
        });
    for (auto& th : pool) th.join();
}

void HfreEncoder::sine_embed(
    const std::vector<std::array<float, 4>>& cxcywh_norm,
    std::vector<float>* out_nxd) const {
    // gen_sineembed_for_position: per component (y, x, w, h), dim_t over
    // pos_embedding_dim/4 frequencies, sin on even / cos on odd indices.
    // The python output concatenates (pos_y, pos_x, pos_w, pos_h) — the input
    // rows are (cx, cy, w, h) but BLOCK 0 = cy and BLOCK 1 = cx.
    const int quarter = cfg_.pos_embedding_dim / 4;
    const int D = cfg_.pos_embedding_dim;
    out_nxd->assign(cxcywh_norm.size() * D, 0.f);
    std::vector<float> dim_t(quarter);
    for (int i = 0; i < quarter; ++i)
        dim_t[i] = std::pow(10000.f, 2.f * float(i / 2) / float(quarter));
    for (size_t n = 0; n < cxcywh_norm.size(); ++n) {
        // comp order in the OUTPUT: y, x, w, h; input rows are (cx, cy, w, h).
        static const int kCompSrc[4] = {1, 0, 2, 3};
        for (int comp = 0; comp < 4; ++comp) {
            const float v = cxcywh_norm[n][kCompSrc[comp]] * 2.f * 3.14159265358979323846f;
            float* dst = &(*out_nxd)[n * D + comp * quarter];
            for (int i = 0; i < quarter; ++i)
                dst[i] = (i % 2 == 0) ? std::sin(v / dim_t[i]) : std::cos(v / dim_t[i]);
        }
    }
}

bool HfreEncoder::encode_aux(const std::vector<Box>& boxes,
                             const std::vector<FeatureLevel>& aux_levels,
                             std::vector<float>* out_aux_nxc,
                             std::string* err) {
    if (boxes.empty()) { out_aux_nxc->clear(); return true; }
    if (aux_levels.empty()) {
        if (err) *err = "no aux feature levels";
        return false;
    }

    // Aux path (no FPN, mm_use_simpleFPN_for_vt_aux=False): all RADIO levels
    // share the grid; concat on channels then one RoIAlign at scale 1/patch.
    const int C0 = aux_levels[0].channels;
    const int H0 = aux_levels[0].height;
    const int W0 = aux_levels[0].width;
    FeatureLevel concat;
    concat.channels = C0 * int(aux_levels.size());
    concat.height = H0;
    concat.width = W0;
    concat.data.resize(size_t(concat.channels) * H0 * W0);
    for (size_t li = 0; li < aux_levels.size(); ++li) {
        if (aux_levels[li].channels != C0 || aux_levels[li].height != H0 ||
            aux_levels[li].width != W0) {
            // Python resizes mismatched levels bilinearly; RADIO levels always
            // share the grid, so a mismatch means a conversion bug.
            if (err) *err = "aux levels must share the same grid";
            return false;
        }
        std::memcpy(&concat.data[size_t(li) * C0 * H0 * W0],
                    aux_levels[li].data.data(),
                    sizeof(float) * aux_levels[li].data.size());
    }
    roi_align_mean_pool(concat, boxes, cfg_.aux_spatial_scale,
                        cfg_.roi_output_size, cfg_.roi_aligned, out_aux_nxc,
                        cfg_.n_threads);
    return true;
}

bool HfreEncoder::encode_vt(const std::vector<Box>& boxes,
                            const std::vector<FeatureLevel>& vt_fpn_levels,
                            const std::vector<float>& feature_strides,
                            std::vector<float>* out_vt_nxc,
                            std::string* err) {
    if (boxes.empty()) { out_vt_nxc->clear(); return true; }
    if (vt_fpn_levels.empty() || vt_fpn_levels.size() != feature_strides.size()) {
        if (err) *err = "vt FPN levels/strides mismatch";
        return false;
    }
    const int C = vt_fpn_levels[0].channels;
    const int N = int(boxes.size());
    const int C_total = C * int(vt_fpn_levels.size());
    out_vt_nxc->assign(size_t(N) * C_total, 0.f);
    for (size_t li = 0; li < vt_fpn_levels.size(); ++li) {
        std::vector<float> level_roi;
        // Python: current_spatial_scale = 1 / feature_strides[level_idx].
        roi_align_mean_pool(vt_fpn_levels[li], boxes,
                            1.f / feature_strides[li],
                            cfg_.roi_output_size, cfg_.roi_aligned, &level_roi,
                            cfg_.n_threads);
        // torch.cat(feats, dim=1): level block INTERLEAVED inside each row —
        // out[n, li*C + ch] (NOT a level-major memcpy of (N, C) blocks).
        for (size_t bi = 0; bi < boxes.size(); ++bi)
            std::memcpy(&(*out_vt_nxc)[bi * C_total + li * C],
                        &level_roi[bi * C], sizeof(float) * C);
    }
    return true;
}

bool HfreEncoder::fuse(const std::vector<float>& aux_nxc,
                       const std::vector<float>& vt_nxc,
                       const std::vector<Box>& boxes,
                       int aux_channels, int vt_channels,
                       const std::vector<FeatureLevel>& vt_grid_ref,
                       std::vector<float>* out_object_features,
                       std::string* err) {
    const int N = int(boxes.size());
    const int D = cfg_.pos_embedding_dim;
    if (aux_channels + vt_channels != D) {
        if (err) *err = "concat dims must equal pos_embedding_dim";
        return false;
    }
    out_object_features->assign(size_t(N) * D, 0.f);
    for (int n = 0; n < N; ++n) {
        std::memcpy(&(*out_object_features)[size_t(n) * D],
                    &aux_nxc[size_t(n) * aux_channels],
                    sizeof(float) * aux_channels);
        std::memcpy(&(*out_object_features)[size_t(n) * D + aux_channels],
                    &vt_nxc[size_t(n) * vt_channels],
                    sizeof(float) * vt_channels);
    }

    // Position embedding: vt branch defines the frame (use_vision_tower_object_feature
    // with plain "concat"); original image extent = finest vt grid / scale.
    if (cfg_.add_pos_embedding && !vt_grid_ref.empty()) {
        // Finest level (index 0) carries the smallest stride.
        const FeatureLevel& finest = vt_grid_ref[0];
        std::vector<std::array<float, 4>> pos_norm(N);
        for (int n = 0; n < N; ++n) {
            // The caller passes vt_grid_ref levels whose pixels are already in
            // original-image scale terms via their strides; the finest level's
            // extent in original pixels is height*stride_of_level0.
            const float img_w = float(finest.width);   // grid_ref pre-scaled by caller
            const float img_h = float(finest.height);
            float x1 = boxes[n].x1 / img_w, x2 = boxes[n].x2 / img_w;
            float y1 = boxes[n].y1 / img_h, y2 = boxes[n].y2 / img_h;
            pos_norm[n] = {(x1 + x2) / 2.f, (y1 + y2) / 2.f, x2 - x1, y2 - y1};
        }
        std::vector<float> pos_emb;
        sine_embed(pos_norm, &pos_emb);
        for (size_t i = 0; i < out_object_features->size(); ++i)
            (*out_object_features)[i] += pos_emb[i];
    }
    return true;
}

}  // namespace vlx
