#include "pillar_vfe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>

namespace openpcdet {

namespace {

template <class F>
void parallel_voxels(int32_t count, F && body) {
    if (count <= 0) return;
    unsigned threads = std::min<unsigned>(std::max(1u, std::thread::hardware_concurrency()),
                                          (unsigned) count);
    if (threads == 1 || count < 256) {
        body(0, count);
        return;
    }
    const int32_t chunk = (count + (int32_t) threads - 1) / (int32_t) threads;
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int32_t begin = 0; begin < count; begin += chunk) {
        const int32_t end = std::min(count, begin + chunk);
        workers.emplace_back([&body, begin, end] { body(begin, end); });
    }
    for (auto & worker : workers) worker.join();
}

// Read a weight tensor as a float vector. Handles f32 and f16; quantized
// types are not supported for VFE weights (the converter always writes them
// as f32 — they're tiny Linear layers, no benefit from quantization). The
// backbone/head conv weights are handled by the ggml backend natively.
std::vector<float> to_f32(const ggml_tensor * t) {
    if (!t) return {};
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t) n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t) n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, out.data(), n);
    } else {
        // Quantized types (q8_0, etc.): use ggml's public type traits
        // to dequantize.  VFE weights are small and only used on CPU.
        const ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (!traits || !traits->to_float) {
            GGML_ASSERT(false && "VFE weights: unsupported quantization type");
        }
        const int64_t row_size = t->ne[0];
        const int64_t rows = n / row_size;
        const size_t src_row_bytes = ggml_row_size(t->type, row_size);
        for (int64_t r = 0; r < rows; ++r) {
            traits->to_float(
                static_cast<const uint8_t *>(t->data) + r * src_row_bytes,
                out.data() + r * row_size, row_size);
        }
    }
    return out;
}

} // namespace

bool load_pillar_vfe_weights(model_file & file, const hparams & hp,
                             pillar_vfe_weights & w, std::string & err) {
    const int n_layers = (int) hp.vfe_num_filters.size();
    if (n_layers == 0) { err = "vfe_num_filters empty"; return false; }

    // input channels to the first PFNLayer = num_point_features + 6 (if use_absolute_xyz)
    // or num_point_features - 3 + 6 (otherwise). Matches PillarVFE.__init__.
    int in_c = hp.num_point_features;
    in_c += hp.use_absolute_xyz ? 6 : 3;
    if (hp.with_distance) in_c += 1;

    w.layers.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        // PFNLayer halves non-final outputs, then concatenates the per-pillar
        // max so the next layer still sees NUM_FILTERS[i] channels.
        const int configured_out_c = hp.vfe_num_filters[i];
        const int out_c = (i == n_layers - 1) ? configured_out_c : configured_out_c / 2;
        if (out_c <= 0) { err = "invalid non-final VFE filter count"; return false; }
        const std::string p = "vfe.pfn." + std::to_string(i) + ".";
        auto & L = w.layers[i];
        L.out_channels = out_c;
        L.last_layer   = (i == n_layers - 1);

        L.linear_w = file.require((p + "linear.weight").c_str());
        if (hp.vfe_use_norm) {
            L.linear_b = nullptr;
            L.bn_scale = file.require((p + "bn_fold.scale").c_str());
            L.bn_bias  = file.require((p + "bn_fold.bias").c_str());
        } else {
            L.linear_b = file.require((p + "linear.bias").c_str());
            L.bn_scale = nullptr;
            L.bn_bias  = nullptr;
        }
        if (!file.error.empty()) { err = file.error; return false; }
        if (L.linear_w->ne[0] != in_c || L.linear_w->ne[1] != out_c) {
            err = "VFE linear weight shape mismatch at layer " + std::to_string(i);
            return false;
        }
        in_c = L.last_layer ? out_c : out_c * 2;
    }
    return true;
}

bool run_pillar_vfe_cpu(const pillar_vfe_weights & w,
                        const hparams & hp,
                        const float * voxels, const int32_t * num_points,
                        const int32_t * coords, int32_t V, int32_t P, int32_t point_dim,
                        std::vector<float> & out_pillar_features,
                        std::string & err,
                        const tap_sink & sink) {
    if (V <= 0 || P <= 0) { err = "empty voxels"; return false; }

    const float vx = hp.voxel_size[0], vy = hp.voxel_size[1], vz = hp.voxel_size[2];
    const float x_off = vx / 2 + hp.point_cloud_range[0];
    const float y_off = vy / 2 + hp.point_cloud_range[1];
    const float z_off = vz / 2 + hp.point_cloud_range[2];

    // --- 1. Expand features: [voxel_features, f_cluster, f_center] ---
    // f_cluster = xyz - mean(xyz over valid points)
    // f_center  = xyz - (coord * voxel_size + offset)
    const bool use_abs = hp.use_absolute_xyz;
    const int  xyz_c   = 3;  // first 3 channels of voxel_features are x,y,z
    const int  in_c    = use_abs ? point_dim + 6 : (point_dim - 3) + 6;
    // expanded layout per voxel, per point: [in_c]
    std::vector<float> feat((size_t) V * P * in_c, 0.0f);

    parallel_voxels(V, [&](int32_t v_begin, int32_t v_end) {
    for (int v = v_begin; v < v_end; v++) {
        const int   np   = num_points[v];
        const float * vf = voxels + (size_t) v * P * point_dim;
        // mean of xyz over valid points
        float mx = 0, my = 0, mz = 0;
        for (int p = 0; p < np; p++) {
            mx += vf[p * point_dim + 0];
            my += vf[p * point_dim + 1];
            mz += vf[p * point_dim + 2];
        }
        const float inv = np > 0 ? 1.0f / np : 0.0f;
        mx *= inv; my *= inv; mz *= inv;

        const int32_t gx = coords[v * 4 + 3];
        const int32_t gy = coords[v * 4 + 2];
        const int32_t gz = coords[v * 4 + 1];
        const float cx = gx * vx + x_off;
        const float cy = gy * vy + y_off;
        const float cz = gz * vz + z_off;

        for (int p = 0; p < P; p++) {
            float * dst = feat.data() + ((size_t) v * P + p) * in_c;
            const float * src = vf + (size_t) p * point_dim;
            const bool valid = (p < np);
            int oc = 0;
            if (use_abs) {
                for (int c = 0; c < point_dim; c++) dst[oc++] = valid ? src[c] : 0.0f;
            } else {
                // drop xyz from voxel_features
                for (int c = xyz_c; c < point_dim; c++) dst[oc++] = valid ? src[c] : 0.0f;
            }
            // f_cluster
            dst[oc++] = valid ? (src[0] - mx) : 0.0f;
            dst[oc++] = valid ? (src[1] - my) : 0.0f;
            dst[oc++] = valid ? (src[2] - mz) : 0.0f;
            // f_center
            dst[oc++] = valid ? (src[0] - cx) : 0.0f;
            dst[oc++] = valid ? (src[1] - cy) : 0.0f;
            dst[oc++] = valid ? (src[2] - cz) : 0.0f;
            if (hp.with_distance) {
                dst[oc++] = valid ? std::sqrt(src[0]*src[0] + src[1]*src[1] + src[2]*src[2]) : 0.0f;
            }
        }
    }
    });

    // --- 2. PFNLayer sequence ---
    // Layout during forward: [V, P, C] (V outermost). We reshape as needed.
    int cur_c = in_c;
    std::vector<float> x = std::move(feat);  // [V*P*in_c]

    for (size_t li = 0; li < w.layers.size(); li++) {
        const auto & L = w.layers[li];
        const int out_c = L.out_channels;

        // Linear: y[v,p,o] = sum_i x[v,p,i] * W[o,i] (+ b[o] if no norm)
        // W is stored as [out_c, in_c] row-major (PyTorch Linear convention).
        const std::vector<float> W   = to_f32(L.linear_w);
        const std::vector<float> B   = L.linear_b ? to_f32(L.linear_b) : std::vector<float>();
        const std::vector<float> SCL = L.bn_scale ? to_f32(L.bn_scale) : std::vector<float>();
        const std::vector<float> BIA = L.bn_bias  ? to_f32(L.bn_bias)  : std::vector<float>();

        std::vector<float> lin((size_t) V * P * out_c, 0.0f);
        parallel_voxels(V, [&](int32_t v_begin, int32_t v_end) {
        for (int v = v_begin; v < v_end; v++) {
            for (int p = 0; p < P; p++) {
                const float * xv = x.data() + ((size_t) v * P + p) * cur_c;
                float * yv = lin.data() + ((size_t) v * P + p) * out_c;
                for (int o = 0; o < out_c; o++) {
                    const float * wrow = W.data() + (size_t) o * cur_c;
                    float acc = 0.0f;
                    for (int i = 0; i < cur_c; i++) acc += xv[i] * wrow[i];
                    if (!B.empty()) acc += B[o];
                    yv[o] = acc;
                }
            }
        }
        });
        if (sink) sink("vfe.layer_" + std::to_string(li) + ".linear",
                       lin.data(), (int64_t) V * P, out_c);

        // BN-folded affine (per-channel)
        if (!SCL.empty()) {
            parallel_voxels(V, [&](int32_t v_begin, int32_t v_end) {
            for (int v = v_begin; v < v_end; v++) {
                float * rows = lin.data() + (size_t) v * P * out_c;
                for (int p = 0; p < P; p++) {
                    for (int o = 0; o < out_c; o++) {
                        rows[(size_t) p * out_c + o] =
                            rows[(size_t) p * out_c + o] * SCL[o] + BIA[o];
                    }
                }
            }
            });
            if (sink) sink("vfe.layer_" + std::to_string(li) + ".bn",
                           lin.data(), (int64_t) V * P, out_c);
        }

        // ReLU
        parallel_voxels(V, [&](int32_t v_begin, int32_t v_end) {
            float * begin = lin.data() + (size_t) v_begin * P * out_c;
            float * end = lin.data() + (size_t) v_end * P * out_c;
            for (float * value = begin; value != end; ++value) *value = std::max(*value, 0.0f);
        });
        if (sink) sink("vfe.layer_" + std::to_string(li) + ".relu",
                       lin.data(), (int64_t) V * P, out_c);

        // Max over P dim -> [V, out_c]
        std::vector<float> x_max((size_t) V * out_c, 0.0f);
        parallel_voxels(V, [&](int32_t v_begin, int32_t v_end) {
        for (int v = v_begin; v < v_end; v++) {
            for (int o = 0; o < out_c; o++) {
                float m = -1e30f;
                for (int p = 0; p < P; p++) {
                    const float val = lin[((size_t) v * P + p) * out_c + o];
                    if (val > m) m = val;
                }
                x_max[(size_t) v * out_c + o] = m;
            }
        }
        });
        if (sink) sink("vfe.layer_" + std::to_string(li) + ".max",
                       x_max.data(), V, out_c);

        if (L.last_layer) {
            // PillarVFE.forward does `features = features.squeeze()` -> [V, out_c]
            out_pillar_features = std::move(x_max);
            cur_c = out_c;
        } else {
            // concat([lin, x_max.repeat(P,1)], dim=-1) -> [V, P, out_c*2]
            // The repeated max is broadcast across the P dim.
            const int new_c = out_c * 2;
            std::vector<float> nxt((size_t) V * P * new_c, 0.0f);
            parallel_voxels(V, [&](int32_t v_begin, int32_t v_end) {
            for (int v = v_begin; v < v_end; v++) {
                for (int p = 0; p < P; p++) {
                    float * dst = nxt.data() + ((size_t) v * P + p) * new_c;
                    const float * lin_row = lin.data() + ((size_t) v * P + p) * out_c;
                    const float * max_row = x_max.data() + (size_t) v * out_c;
                    for (int o = 0; o < out_c; o++) dst[o] = lin_row[o];
                    for (int o = 0; o < out_c; o++) dst[out_c + o] = max_row[o];
                }
            }
            });
            x = std::move(nxt);
            cur_c = new_c;
        }
    }

    if (sink) sink("vfe.out", out_pillar_features.data(), V, cur_c);
    return true;
}

bool run_dynamic_pillar_vfe_cpu(const pillar_vfe_weights & w,
                                const hparams & hp,
                                const float * points, int32_t point_count,
                                int32_t point_dim,
                                std::vector<int32_t> & pillar_coords,
                                std::vector<float> & pillar_features,
                                std::string & err,
                                const tap_sink & sink) {
    pillar_coords.clear();
    pillar_features.clear();
    if (!points || point_count <= 0 || point_dim < 3 || hp.grid_size.size() < 2) {
        err = "bad dynamic-pillar input";
        return false;
    }
    const int nx = hp.grid_size[0], ny = hp.grid_size[1];
    const float vx = hp.voxel_size[0], vy = hp.voxel_size[1], vz = hp.voxel_size[2];
    const float x0 = hp.point_cloud_range[0], y0 = hp.point_cloud_range[1];
    struct point_entry { int32_t source, gx, gy, group; };
    std::vector<point_entry> entries;
    entries.reserve(point_count);
    // A dense cell-to-group table is only nx*ny integers (1 MiB for the
    // nuScenes 512x512 grid). Scanning it in key order preserves the original
    // gx-major sorted pillar order while replacing sort + P binary searches
    // with linear O(P + nx*ny) work.
    std::vector<int32_t> group_by_key(static_cast<size_t>(nx) * ny, -1);
    for (int32_t point = 0; point < point_count; point++) {
        const float * value = points + static_cast<int64_t>(point) * point_dim;
        const int gx = static_cast<int>(std::floor((value[0] - x0) / vx));
        const int gy = static_cast<int>(std::floor((value[1] - y0) / vy));
        if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) continue;
        entries.push_back({point, gx, gy, -1});
        group_by_key[static_cast<size_t>(gx) * ny + gy] = -2;
    }
    if (entries.empty()) return true;
    int32_t group_count = 0;
    pillar_coords.reserve(std::min<size_t>(entries.size(), group_by_key.size()) * 4);
    for (size_t key = 0; key < group_by_key.size(); ++key) {
        if (group_by_key[key] != -2) continue;
        group_by_key[key] = group_count++;
        const int32_t gx = static_cast<int32_t>(key / ny);
        const int32_t gy = static_cast<int32_t>(key % ny);
        pillar_coords.insert(pillar_coords.end(), {0, 0, gy, gx});
    }
    for (point_entry & entry : entries) {
        entry.group = group_by_key[static_cast<size_t>(entry.gx) * ny + entry.gy];
    }

    std::vector<float> means(static_cast<size_t>(group_count) * 3, 0.0f);
    std::vector<int32_t> counts(group_count, 0);
    for (const point_entry & entry : entries) {
        const float * point = points + static_cast<int64_t>(entry.source) * point_dim;
        float * mean = means.data() + static_cast<size_t>(entry.group) * 3;
        mean[0] += point[0]; mean[1] += point[1]; mean[2] += point[2];
        counts[entry.group]++;
    }
    for (int32_t group = 0; group < group_count; group++) {
        const float inverse = 1.0f / counts[group];
        for (int axis = 0; axis < 3; axis++) means[group * 3 + axis] *= inverse;
    }

    const bool use_abs = hp.use_absolute_xyz;
    const int input_channels = (use_abs ? point_dim : point_dim - 3) + 6 +
                               (hp.with_distance ? 1 : 0);
    const int64_t active_points = static_cast<int64_t>(entries.size());
    std::vector<float> features(static_cast<size_t>(active_points) * input_channels, 0.0f);
    const float x_offset = vx * 0.5f + x0;
    const float y_offset = vy * 0.5f + y0;
    const float z_offset = vz * 0.5f + hp.point_cloud_range[2];
    for (int64_t row = 0; row < active_points; row++) {
        const point_entry & entry = entries[row];
        const float * point = points + static_cast<int64_t>(entry.source) * point_dim;
        const float * mean = means.data() + static_cast<size_t>(entry.group) * 3;
        float * dst = features.data() + row * input_channels;
        int channel = 0;
        if (use_abs) {
            for (int source = 0; source < point_dim; source++) dst[channel++] = point[source];
        } else {
            for (int source = 3; source < point_dim; source++) dst[channel++] = point[source];
        }
        dst[channel++] = point[0] - mean[0];
        dst[channel++] = point[1] - mean[1];
        dst[channel++] = point[2] - mean[2];
        dst[channel++] = point[0] - (entry.gx * vx + x_offset);
        dst[channel++] = point[1] - (entry.gy * vy + y_offset);
        dst[channel++] = point[2] - z_offset;
        if (hp.with_distance) dst[channel++] =
            std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
    }

    int current_channels = input_channels;
    for (size_t layer_index = 0; layer_index < w.layers.size(); layer_index++) {
        const pfn_layer_weights & layer = w.layers[layer_index];
        const int output_channels = layer.out_channels;
        const std::vector<float> weight = to_f32(layer.linear_w);
        const std::vector<float> bias = layer.linear_b ? to_f32(layer.linear_b) : std::vector<float>();
        const std::vector<float> scale = layer.bn_scale ? to_f32(layer.bn_scale) : std::vector<float>();
        const std::vector<float> bn_bias = layer.bn_bias ? to_f32(layer.bn_bias) : std::vector<float>();
        std::vector<float> linear(static_cast<size_t>(active_points) * output_channels, 0.0f);
        parallel_voxels(static_cast<int32_t>(active_points), [&](int32_t begin, int32_t end) {
            for (int32_t row = begin; row < end; row++) {
                const float * src = features.data() + static_cast<int64_t>(row) * current_channels;
                float * dst = linear.data() + static_cast<int64_t>(row) * output_channels;
                for (int output_channel = 0; output_channel < output_channels; output_channel++) {
                    const float * weight_row = weight.data() +
                        static_cast<size_t>(output_channel) * current_channels;
                    float value = bias.empty() ? 0.0f : bias[output_channel];
                    for (int input_channel = 0; input_channel < current_channels; input_channel++)
                        value += src[input_channel] * weight_row[input_channel];
                    if (!scale.empty()) value = value * scale[output_channel] + bn_bias[output_channel];
                    dst[output_channel] = std::max(value, 0.0f);
                }
            }
        });
        if (sink) sink("vfe.layer_" + std::to_string(layer_index) + ".relu",
                       linear.data(), active_points, output_channels);
        std::vector<float> maxima(static_cast<size_t>(group_count) * output_channels,
                                  -1e30f);
        for (int64_t row = 0; row < active_points; row++) {
            float * dst = maxima.data() + static_cast<size_t>(entries[row].group) * output_channels;
            const float * src = linear.data() + row * output_channels;
            for (int channel = 0; channel < output_channels; channel++)
                dst[channel] = std::max(dst[channel], src[channel]);
        }
        if (sink) sink("vfe.layer_" + std::to_string(layer_index) + ".max",
                       maxima.data(), group_count, output_channels);
        if (layer.last_layer) {
            pillar_features = std::move(maxima);
        } else {
            features.resize(static_cast<size_t>(active_points) * output_channels * 2);
            for (int64_t row = active_points; row-- > 0;) {
                float * dst = features.data() + row * output_channels * 2;
                const float * local = linear.data() + row * output_channels;
                const float * aggregate = maxima.data() +
                    static_cast<size_t>(entries[row].group) * output_channels;
                std::copy_n(local, output_channels, dst);
                std::copy_n(aggregate, output_channels, dst + output_channels);
            }
            current_channels = output_channels * 2;
        }
    }
    if (sink && !pillar_features.empty())
        sink("vfe.out", pillar_features.data(), group_count, w.layers.back().out_channels);
    return true;
}

} // namespace openpcdet
