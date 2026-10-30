#include "mape_pillar_vfe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace openpcdet {

namespace {

// Read weight tensor as float vector (handles f32/f16)
std::vector<float> to_f32(const ggml_tensor * t) {
    if (!t) return {};
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t) n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t) n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, out.data(), n);
    } else {
        const ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            const int64_t row_size = t->ne[0];
            const int64_t rows = n / row_size;
            const size_t src_row_bytes = ggml_row_size(t->type, row_size);
            for (int64_t r = 0; r < rows; ++r) {
                traits->to_float(
                    static_cast<const uint8_t *>(t->data) + r * src_row_bytes,
                    out.data() + r * row_size, row_size);
            }
        }
    }
    return out;
}

// Simple voxelization: group points by (x, y) pillar, keep up to max_points per pillar
struct VoxelData {
    std::vector<std::vector<float>> points;  // up to max_points
};

struct VoxelizationResult {
    std::unordered_map<int64_t, VoxelData> voxels;
    std::vector<std::pair<int, int>> pillar_coords;  // (grid_y, grid_x) per voxel
    int grid_x = 0, grid_y = 0;
};

VoxelizationResult voxelize_pillars(const float * points, int n_points, int point_dim,
                                    const hparams & hp, int max_points = 32) {
    VoxelizationResult result;

    float vx = hp.voxel_size[0], vy = hp.voxel_size[1];
    float range_x0 = hp.point_cloud_range[0];
    float range_y0 = hp.point_cloud_range[1];
    float range_x1 = hp.point_cloud_range[3];
    float range_y1 = hp.point_cloud_range[4];

    result.grid_x = (int) ((range_x1 - range_x0) / vx);
    result.grid_y = (int) ((range_y1 - range_y0) / vy);

    for (int i = 0; i < n_points; i++) {
        const float * p = points + i * point_dim;
        float x = p[0], y = p[1];
        int gx = (int) ((x - range_x0) / vx);
        int gy = (int) ((y - range_y0) / vy);
        if (gx < 0 || gx >= result.grid_x || gy < 0 || gy >= result.grid_y) continue;

        int64_t key = (int64_t) gy * result.grid_x + gx;
        auto & voxel = result.voxels[key];
        if ((int) voxel.points.size() < max_points) {
            voxel.points.emplace_back(p, p + point_dim);
        }
    }

    // Extract sorted pillar coordinates
    result.pillar_coords.reserve(result.voxels.size());
    for (auto & [key, _] : result.voxels) {
        int gy = (int) (key / result.grid_x);
        int gx = (int) (key % result.grid_x);
        result.pillar_coords.emplace_back(gy, gx);
    }
    std::sort(result.pillar_coords.begin(), result.pillar_coords.end());

    return result;
}

} // anon ns

bool load_mape_pillar_weights(model_file & file, const hparams & hp,
                              mape_pillar_weights & w, std::string & error) {
    // Compute input channels: point_features + 6 (cluster + center offsets)
    int point_features = hp.num_point_features;
    w.in_channels = point_features + 6;  // xyz + intensity/ts + cluster(3) + center(3)

    int num_layers = (int) hp.vfe_num_filters.size();
    if (num_layers == 0) {
        error = "mape_pillar: no VFE filters configured";
        return false;
    }

    w.layers.resize(num_layers);
    int in_ch = w.in_channels;
    for (int i = 0; i < num_layers; i++) {
        auto & layer = w.layers[i];
        int out_ch = hp.vfe_num_filters[i];

        char buf[256];
        snprintf(buf, sizeof(buf), "vfe.mape_layers.%d.linear.weight", i);
        layer.linear_w = file.require(buf);
        snprintf(buf, sizeof(buf), "vfe.mape_layers.%d.linear.bn_fold.scale", i);
        layer.bn_scale = file.require(buf);
        snprintf(buf, sizeof(buf), "vfe.mape_layers.%d.linear.bn_fold.bias", i);
        layer.bn_bias = file.require(buf);
        snprintf(buf, sizeof(buf), "vfe.mape_layers.%d.attn_mlp.0.weight", i);
        layer.attn_fc1_w = file.require(buf);
        snprintf(buf, sizeof(buf), "vfe.mape_layers.%d.attn_mlp.0.bias", i);
        layer.attn_fc1_b = file.require(buf);
        snprintf(buf, sizeof(buf), "vfe.mape_layers.%d.attn_mlp.2.weight", i);
        layer.attn_fc2_w = file.require(buf);
        snprintf(buf, sizeof(buf), "vfe.mape_layers.%d.attn_mlp.2.bias", i);
        layer.attn_fc2_b = file.require(buf);

        if (!file.error.empty()) { error = file.error; return false; }
        in_ch = out_ch;
    }
    w.out_channels = hp.vfe_num_filters.back();
    return true;
}

bool run_mape_pillar_vfe(engine_backend & /*backend*/, const hparams & hp,
                         const mape_pillar_weights & w,
                         const float * points, int n_points, int point_dim,
                         mape_pillar_result & result,
                         const tap_sink & /*sink*/, std::string & error) {
    if (n_points <= 0 || !points) {
        result.num_pillars = 0;
        return true;
    }

    const int max_points = 32;
    auto vox = voxelize_pillars(points, n_points, point_dim, hp, max_points);

    int num_pillars = (int) vox.pillar_coords.size();
    result.num_pillars = num_pillars;
    result.width = vox.grid_x;
    result.height = vox.grid_y;

    if (num_pillars == 0) return true;

    int out_ch = w.out_channels;
    result.pillar_features.resize((size_t) num_pillars * out_ch);
    result.coords.resize((size_t) num_pillars * 3);

    // Pre-load weights as float vectors for CPU computation
    struct LayerWeights {
        std::vector<float> linear_w, bn_scale, bn_bias;
        std::vector<float> attn_fc1_w, attn_fc1_b, attn_fc2_w, attn_fc2_b;
        int in_ch, out_ch;
    };
    std::vector<LayerWeights> lw(w.layers.size());
    for (size_t i = 0; i < w.layers.size(); i++) {
        lw[i].linear_w = to_f32(w.layers[i].linear_w);
        lw[i].bn_scale = to_f32(w.layers[i].bn_scale);
        lw[i].bn_bias  = to_f32(w.layers[i].bn_bias);
        lw[i].attn_fc1_w = to_f32(w.layers[i].attn_fc1_w);
        lw[i].attn_fc1_b = to_f32(w.layers[i].attn_fc1_b);
        lw[i].attn_fc2_w = to_f32(w.layers[i].attn_fc2_w);
        lw[i].attn_fc2_b = to_f32(w.layers[i].attn_fc2_b);
        lw[i].in_ch = (int) lw[i].linear_w.size() / (int) lw[i].bn_scale.size();
        lw[i].out_ch = (int) lw[i].bn_scale.size();
    }

    float vx = hp.voxel_size[0], vy = hp.voxel_size[1], vz = hp.voxel_size[2];
    float rx0 = hp.point_cloud_range[0], ry0 = hp.point_cloud_range[1], rz0 = hp.point_cloud_range[2];

    for (int pi = 0; pi < num_pillars; pi++) {
        auto [gy, gx] = vox.pillar_coords[pi];
        int64_t key = (int64_t) gy * vox.grid_x + gx;
        auto & voxel_data = vox.voxels[key];
        int M = (int) voxel_data.points.size();

        // Voxel center
        float cx = gx * vx + vx / 2.0f + rx0;
        float cy = gy * vy + vy / 2.0f + ry0;

        // Compute cluster center (mean of points)
        float mean_x = 0, mean_y = 0, mean_z = 0;
        for (auto & p : voxel_data.points) {
            mean_x += p[0]; mean_y += p[1]; mean_z += p[2];
        }
        mean_x /= M; mean_y /= M; mean_z /= M;

        // Build augmented features: [M, in_ch]
        int in_ch = lw[0].in_ch;
        std::vector<float> features((size_t) M * in_ch);
        for (int m = 0; m < M; m++) {
            auto & p = voxel_data.points[m];
            // Original features
            for (int d = 0; d < point_dim && d < in_ch; d++)
                features[m * in_ch + d] = p[d];
            // Cluster offsets (xyz - mean)
            if (in_ch > point_dim + 0) {
                features[m * in_ch + point_dim + 0] = p[0] - mean_x;
                features[m * in_ch + point_dim + 1] = p[1] - mean_y;
                features[m * in_ch + point_dim + 2] = p[2] - mean_z;
            }
            // Center offsets (xyz - voxel_center)
            if (in_ch > point_dim + 3) {
                features[m * in_ch + point_dim + 3] = p[0] - cx;
                features[m * in_ch + point_dim + 4] = p[1] - cy;
                features[m * in_ch + point_dim + 5] = p[2] - (rz0 + vz / 2.0f);
            }
        }

        // Process through MAPE layers
        std::vector<float> pillar_feat;
        for (size_t li = 0; li < lw.size(); li++) {
            auto & layer = lw[li];
            int cur_in = layer.in_ch;
            int cur_out = layer.out_ch;

            // Linear + BN: y = scale * (W^T x) + bias
            std::vector<float> projected((size_t) M * cur_out);
            for (int m = 0; m < M; m++) {
                for (int o = 0; o < cur_out; o++) {
                    float val = 0;
                    for (int j = 0; j < cur_in; j++) {
                        // Weight layout: [cur_in, cur_out] (transposed from PyTorch)
                        val += features[m * cur_in + j] * layer.linear_w[j * cur_out + o];
                    }
                    // BN fold: y = scale * x + bias
                    val = layer.bn_scale[o] * val + layer.bn_bias[o];
                    // ReLU
                    projected[m * cur_out + o] = std::max(0.0f, val);
                }
            }

            // Max pooling
            std::vector<float> f_max(cur_out, -1e30f);
            for (int m = 0; m < M; m++)
                for (int o = 0; o < cur_out; o++)
                    f_max[o] = std::max(f_max[o], projected[m * cur_out + o]);
            for (int o = 0; o < cur_out; o++)
                f_max[o] = std::max(0.0f, f_max[o]);  // clamp for empty pillars

            // Attention pooling
            // attn_scores = attn_mlp(projected).sum(-1) → softmax → weighted sum
            std::vector<float> attn_scores(M);
            for (int m = 0; m < M; m++) {
                // FC1 + ReLU
                std::vector<float> hidden(cur_out);
                for (int o = 0; o < cur_out; o++) {
                    float val = layer.attn_fc1_b[o];
                    for (int j = 0; j < cur_out; j++)
                        val += projected[m * cur_out + j] * layer.attn_fc1_w[j * cur_out + o];
                    hidden[o] = std::max(0.0f, val);
                }
                // FC2 + sum
                float score = layer.attn_fc2_b[0];
                for (int j = 0; j < cur_out; j++)
                    score += hidden[j] * layer.attn_fc2_w[j * cur_out];
                // Actually sum over all output channels
                score = 0;
                for (int o = 0; o < cur_out; o++) {
                    float val = layer.attn_fc2_b[o];
                    for (int j = 0; j < cur_out; j++)
                        val += hidden[j] * layer.attn_fc2_w[j * cur_out + o];
                    score += val;
                }
                attn_scores[m] = score;
            }
            // Softmax
            float max_s = *std::max_element(attn_scores.begin(), attn_scores.end());
            float sum_exp = 0;
            for (auto & s : attn_scores) { s = std::exp(s - max_s); sum_exp += s; }
            for (auto & s : attn_scores) s /= sum_exp;

            // Weighted sum
            std::vector<float> f_attn(cur_out, 0);
            for (int m = 0; m < M; m++)
                for (int o = 0; o < cur_out; o++)
                    f_attn[o] += attn_scores[m] * projected[m * cur_out + o];

            // Average of max and attention
            pillar_feat.resize(cur_out);
            for (int o = 0; o < cur_out; o++)
                pillar_feat[o] = (f_max[o] + f_attn[o]) / 2.0f;

            // Feed to next layer
            features = pillar_feat;
            features.resize((size_t) M * cur_out);
            for (int m = 1; m < M; m++)
                for (int o = 0; o < cur_out; o++)
                    features[m * cur_out + o] = pillar_feat[o];
        }

        // Store result
        for (int o = 0; o < out_ch; o++)
            result.pillar_features[(size_t) pi * out_ch + o] = pillar_feat[o];

        // Coords: (batch_idx=0, y=gy, x=gx)
        result.coords[(size_t) pi * 3 + 0] = 0;
        result.coords[(size_t) pi * 3 + 1] = gy;
        result.coords[(size_t) pi * 3 + 2] = gx;
    }

    return true;
}

} // namespace openpcdet
