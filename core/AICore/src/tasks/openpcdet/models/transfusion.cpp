#include "transfusion.h"

#include "voxel_backbone_runner.h"

#include <ggml.h>
#include <chrono>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace openpcdet {
namespace {

ggml_tensor * require(model_file & file, const std::string & name) {
    return file.require(name.c_str());
}

bool load_prediction(model_file & file, const std::string & name,
                     transfusion_prediction_weights & output) {
    const std::string prefix = "head.tf.pred." + name;
    output.hidden.conv_w = require(file, prefix + ".0.0.weight");
    output.hidden.bn_scale = require(file, prefix + ".0.1.eval_scale");
    output.hidden.bn_bias = require(file, prefix + ".0.1.eval_bias");
    output.output_weight = require(file, prefix + ".1.weight");
    output.output_bias = require(file, prefix + ".1.bias");
    return file.error.empty();
}

void remap_tensor(ggml_tensor *& tensor, ggml_context * context) {
    if (tensor && context) tensor = ggml_get_tensor(context, ggml_get_name(tensor));
}

ggml_tensor * linear(ggml_context * context, ggml_tensor * weight,
                     ggml_tensor * bias, ggml_tensor * input) {
    ggml_tensor * matrix = ggml_reshape_2d(
        context, weight, input->ne[0], ggml_nelements(weight) / input->ne[0]);
    ggml_tensor * output = ggml_mul_mat(context, matrix, input);
    return bias ? ggml_add(context, output, bias) : output;
}

ggml_tensor * affine_norm(ggml_context * context, ggml_tensor * input,
                          ggml_tensor * scale, ggml_tensor * bias) {
    return ggml_add(context, ggml_mul(context, ggml_norm(context, input, 1e-5f), scale), bias);
}

ggml_tensor * position_embedding(ggml_context * context, ggml_tensor * positions,
                                 ggml_tensor * weight1, ggml_tensor * bias1,
                                 ggml_tensor * scale, ggml_tensor * shift,
                                 ggml_tensor * weight2, ggml_tensor * bias2) {
    ggml_tensor * output = linear(context, weight1, bias1, positions);
    output = ggml_add(context, ggml_mul(context, output, scale), shift);
    output = ggml_relu(context, output);
    return linear(context, weight2, bias2, output);
}

ggml_tensor * attention(ggml_context * context,
                        ggml_tensor * q_weight, ggml_tensor * q_bias,
                        ggml_tensor * k_weight, ggml_tensor * k_bias,
                        ggml_tensor * v_weight, ggml_tensor * v_bias,
                        ggml_tensor * out_weight, ggml_tensor * out_bias,
                        ggml_tensor * query_input, ggml_tensor * key_input,
                        int heads, ggml_tensor ** probabilities = nullptr,
                        ggml_tensor ** merged_values = nullptr,
                        ggml_tensor ** projected_query = nullptr,
                        ggml_tensor ** projected_key = nullptr,
                        ggml_tensor ** projected_value = nullptr) {
    ggml_tensor * query = linear(context, q_weight, q_bias, query_input);
    ggml_tensor * key = linear(context, k_weight, k_bias, key_input);
    ggml_tensor * value = linear(context, v_weight, v_bias, key_input);
    const int64_t channels = query->ne[0];
    const int64_t head_channels = channels / heads;
    ggml_tensor * q = ggml_permute(context,
        ggml_cont_3d(context, query, head_channels, heads, query->ne[1]), 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(context,
        ggml_reshape_3d(context, key, head_channels, heads, key->ne[1]), 0, 2, 1, 3);
    ggml_tensor * v = ggml_cont_3d(context,
        ggml_permute(context,
            ggml_reshape_3d(context, value, head_channels, heads, value->ne[1]),
            1, 2, 0, 3),
        value->ne[1], head_channels, heads);
    ggml_tensor * scores = ggml_scale(
        context, ggml_mul_mat(context, k, q),
        1.0f / std::sqrt(static_cast<float>(head_channels)));
    scores = ggml_soft_max(context, scores);
    ggml_tensor * attended = ggml_mul_mat(context, v, scores);
    ggml_tensor * joined = ggml_cont_2d(
        context, ggml_permute(context, attended, 0, 2, 1, 3),
        channels, query->ne[1]);
    if (probabilities) *probabilities = scores;
    if (merged_values) *merged_values = joined;
    if (projected_query) *projected_query = query;
    if (projected_key) *projected_key = key;
    if (projected_value) *projected_value = value;
    return linear(context, out_weight, out_bias, joined);
}

ggml_tensor * prediction(ggml_context * context,
                         const transfusion_prediction_weights & weights,
                         ggml_tensor * query) {
    ggml_tensor * hidden = linear(context, weights.hidden.conv_w, nullptr, query);
    hidden = ggml_add(context, ggml_mul(context, hidden, weights.hidden.bn_scale),
                      weights.hidden.bn_bias);
    hidden = ggml_relu(context, hidden);
    return linear(context, weights.output_weight, weights.output_bias, hidden);
}

std::vector<float> readback(ggml_tensor * tensor) {
    std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    return result;
}

std::vector<float> hwc_to_whc(const std::vector<float> & input,
                              int width, int height, int channels) {
    std::vector<float> output(input.size());
    for (int channel = 0; channel < channels; ++channel)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                output[static_cast<size_t>(x) + static_cast<size_t>(width) *
                    (y + static_cast<size_t>(height) * channel)] =
                    input[(static_cast<size_t>(y) * width + x) * channels + channel];
    return output;
}

} // namespace

bool transfusion::load(const std::string & path, const std::string & device, int threads) {
    if (!model::load(path, device, threads)) return false;
    const hparams & hp = file.hp;
    if ((hp.arch != arch_kind::transfusion && hp.arch != arch_kind::bevfusion) ||
        hp.head_kind != dense_head_kind::transfusion ||
        hp.transfusion_hidden_channels % hp.transfusion_num_heads != 0) {
        error = "unsupported TransFusion topology";
        return false;
    }
    if (!load_sparse_backbone_weights(file, hp, weights.sparse, error) ||
        !load_bev_backbone_weights(file, hp, weights.backbone, error)) return false;

    weights.shared_weight = require(file, "head.tf.shared.weight");
    weights.shared_bias = require(file, "head.tf.shared.bias");
    weights.heatmap_hidden.conv_w = require(file, "head.tf.dense_hm.0.conv.weight");
    weights.heatmap_hidden.bn_scale = require(file, "head.tf.dense_hm.0.bn.eval_scale");
    weights.heatmap_hidden.bn_bias = require(file, "head.tf.dense_hm.0.bn.eval_bias");
    weights.heatmap_weight = require(file, "head.tf.dense_hm.1.weight");
    weights.heatmap_bias = require(file, "head.tf.dense_hm.1.bias");
    weights.class_weight = require(file, "head.tf.class.weight");
    weights.class_bias = require(file, "head.tf.class.bias");

    const std::string decoder = "head.tf.dec.";
    weights.self_q_weight = require(file, decoder + "self.q_weight");
    weights.self_q_bias = require(file, decoder + "self.q_bias");
    weights.self_k_weight = require(file, decoder + "self.k_weight");
    weights.self_k_bias = require(file, decoder + "self.k_bias");
    weights.self_v_weight = require(file, decoder + "self.v_weight");
    weights.self_v_bias = require(file, decoder + "self.v_bias");
    weights.self_out_weight = require(file, decoder + "self.out_proj.weight");
    weights.self_out_bias = require(file, decoder + "self.out_proj.bias");
    weights.cross_q_weight = require(file, decoder + "cross.q_weight");
    weights.cross_q_bias = require(file, decoder + "cross.q_bias");
    weights.cross_k_weight = require(file, decoder + "cross.k_weight");
    weights.cross_k_bias = require(file, decoder + "cross.k_bias");
    weights.cross_v_weight = require(file, decoder + "cross.v_weight");
    weights.cross_v_bias = require(file, decoder + "cross.v_bias");
    weights.cross_out_weight = require(file, decoder + "cross.out_proj.weight");
    weights.cross_out_bias = require(file, decoder + "cross.out_proj.bias");
    weights.linear1_weight = require(file, decoder + "linear1.weight");
    weights.linear1_bias = require(file, decoder + "linear1.bias");
    weights.linear2_weight = require(file, decoder + "linear2.weight");
    weights.linear2_bias = require(file, decoder + "linear2.bias");
    weights.norm1_weight = require(file, decoder + "norm1.weight");
    weights.norm1_bias = require(file, decoder + "norm1.bias");
    weights.norm2_weight = require(file, decoder + "norm2.weight");
    weights.norm2_bias = require(file, decoder + "norm2.bias");
    weights.norm3_weight = require(file, decoder + "norm3.weight");
    weights.norm3_bias = require(file, decoder + "norm3.bias");

    auto load_pos = [&](const std::string & name, ggml_tensor *& weight1,
                        ggml_tensor *& bias1, ggml_tensor *& scale, ggml_tensor *& shift,
                        ggml_tensor *& weight2, ggml_tensor *& bias2) {
        const std::string prefix = decoder + name + ".";
        weight1 = require(file, prefix + "0.weight"); bias1 = require(file, prefix + "0.bias");
        scale = require(file, prefix + "1.eval_scale"); shift = require(file, prefix + "1.eval_bias");
        weight2 = require(file, prefix + "3.weight"); bias2 = require(file, prefix + "3.bias");
    };
    load_pos("spos", weights.self_pos_weight1, weights.self_pos_bias1,
             weights.self_pos_scale, weights.self_pos_shift,
             weights.self_pos_weight2, weights.self_pos_bias2);
    load_pos("cpos", weights.cross_pos_weight1, weights.cross_pos_bias1,
             weights.cross_pos_scale, weights.cross_pos_shift,
             weights.cross_pos_weight2, weights.cross_pos_bias2);
    if (!load_prediction(file, "center", weights.center) ||
        !load_prediction(file, "height", weights.height) ||
        !load_prediction(file, "dim", weights.dimensions) ||
        !load_prediction(file, "rot", weights.rotation) ||
        !load_prediction(file, "vel", weights.velocity) ||
        !load_prediction(file, "heatmap", weights.heatmap) || !file.error.empty()) {
        error = file.error;
        return false;
    }

    if (device_ctx) {
        for (auto & layer : weights.sparse.layers) {
            remap_tensor(layer.weight, device_ctx); remap_tensor(layer.bias, device_ctx);
        }
        for (auto & level : weights.backbone.levels) for (auto & block : level) {
            remap_tensor(block.conv_w, device_ctx); remap_tensor(block.bn_scale, device_ctx);
            remap_tensor(block.bn_bias, device_ctx);
        }
        for (auto & block : weights.backbone.deblocks) {
            remap_tensor(block.conv_w, device_ctx); remap_tensor(block.bn_scale, device_ctx);
            remap_tensor(block.bn_bias, device_ctx);
        }
        auto remap_prediction = [&](transfusion_prediction_weights & prediction) {
            remap_tensor(prediction.hidden.conv_w, device_ctx);
            remap_tensor(prediction.hidden.bn_scale, device_ctx);
            remap_tensor(prediction.hidden.bn_bias, device_ctx);
            remap_tensor(prediction.output_weight, device_ctx);
            remap_tensor(prediction.output_bias, device_ctx);
        };
        remap_prediction(weights.center); remap_prediction(weights.height);
        remap_prediction(weights.dimensions); remap_prediction(weights.rotation);
        remap_prediction(weights.velocity); remap_prediction(weights.heatmap);
        ggml_tensor ** begin = &weights.shared_weight;
        ggml_tensor ** end = &weights.cross_pos_bias2 + 1;
        for (ggml_tensor ** tensor = begin; tensor != end; ++tensor)
            remap_tensor(*tensor, device_ctx);
    }
    return true;
}

bool transfusion::forward(const float * points, int32_t count, int32_t dimension,
                          std::vector<detection> & detections, const tap_sink & sink) {
    detections.clear();
    const hparams & hp = file.hp;
    voxel_backbone_result voxel;
    if (!run_voxel_backbone(be, hp, weights.sparse, points, count, dimension,
                            false, voxel, sink, error)) return false;
    if (voxel.spatial_features.empty()) return true;

    return forward_spatial(voxel.spatial_features, voxel.width, voxel.height,
                           detections, sink);
}

bool transfusion::forward_spatial(const std::vector<float> & spatial_features,
                                  int width, int height,
                                  std::vector<detection> & detections,
                                  const tap_sink & sink,
                                  bool input_is_whc) {
    detections.clear();
    const hparams & hp = file.hp;
    if (width <= 0 || height <= 0 ||
        spatial_features.size() != static_cast<size_t>(width) * height * hp.num_bev_features) {
        error = "invalid TransFusion BEV input";
        return false;
    }
    // The ggml graph consumes WHC planes [x + W*y + W*H*c]. Callers holding
    // HWC-interleaved data pay one host transpose; WHC sources (the BevFusion
    // fuser output) skip it entirely.
    const std::vector<float> owned =
        input_is_whc ? std::vector<float>{} : hwc_to_whc(
        spatial_features, width, height, hp.num_bev_features);
    const std::vector<float> & spatial = input_is_whc ? spatial_features : owned;

    const bool profile = false;  // AICore port note: stderr lap profiling dropped
    auto lap_start = std::chrono::steady_clock::now();
    auto lap = [&](const char * name) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[head] %-22s %8.2f ms\n", name,
                     std::chrono::duration<double, std::milli>(now - lap_start).count());
        lap_start = now;
    };
    lap("input_convert");
    // Cross-frame cache for the heatmap graph (see swin_graph_cache rationale):
// identical structure every frame; key stability also enables CUDA-graph reuse.
struct tf_heatmap_cache {
    bool valid = false;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr, * lidar = nullptr, * dense_heatmap = nullptr;
    ggml_tensor * bev = nullptr;
    int W = 0, H = 0, C = 0;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> const_sets;
};
static tf_heatmap_cache g_tf_hm_cache;

constexpr size_t graph_nodes = 8192;
    ggml_init_params params{ggml_tensor_overhead() * graph_nodes +
        ggml_graph_overhead_custom(graph_nodes, false), nullptr, true};
    const bool hm_dump = false;  // AICore port note: geom dump dropped
    // Disabled by default: hit-path segfault under investigation (likely a
    // gallocr/weights-buffer lifetime issue). Opt in via OPCPD_TF_CACHE=1.
    const bool tf_cache_on = false;  // AICore port note: env opt-in cache dropped
    const bool hm_reuse = tf_cache_on && g_tf_hm_cache.valid && !hm_dump
        && g_tf_hm_cache.W == width && g_tf_hm_cache.H == height
        && g_tf_hm_cache.C == hp.num_bev_features;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr, * lidar = nullptr, * dense_heatmap = nullptr;
    ggml_tensor * bev = nullptr;
    if (hm_reuse) {
        context = g_tf_hm_cache.context;
        graph = g_tf_hm_cache.graph;
        input = g_tf_hm_cache.input;
        lidar = nullptr;                // recovered below via graph outputs
        dense_heatmap = nullptr;
        // Outputs are reachable through the graph; keep pointers cached too.
        lidar = g_tf_hm_cache.lidar;
        bev = g_tf_hm_cache.bev;
        dense_heatmap = g_tf_hm_cache.dense_heatmap;
        // Order matches the proven swin hit path: set(input) BEFORE alloc.
        ggml_backend_tensor_set(input, spatial.data(), 0, spatial.size() * sizeof(float));
        if (!ggml_gallocr_alloc_graph(be.galloc, graph)) {
            error = "TransFusion cached heatmap graph alloc failed";
            g_tf_hm_cache = {};
            ggml_free(context);
            return false;
        }
    } else {
    context = ggml_init(params);
    if (!context) { error = "TransFusion heatmap context init failed"; return false; }
    input = ggml_new_tensor_4d(
        context, GGML_TYPE_F32, width, height, hp.num_bev_features, 1);
    ggml_set_input(input);
    bev = build_bev_backbone_graph(context, weights.backbone, hp, input);
    ggml_set_output(bev);
    lidar = build_conv_bias(
        context, weights.shared_weight, weights.shared_bias, bev, 1, 1);
    ggml_tensor * heat_hidden = build_conv_bn_relu(
        context, weights.heatmap_hidden, lidar, 1, 1);
    dense_heatmap = build_conv_bias(
        context, weights.heatmap_weight, weights.heatmap_bias, heat_hidden, 1, 1);
    ggml_set_output(lidar); ggml_set_output(dense_heatmap);
    graph = ggml_new_graph_custom(context, graph_nodes, false);
    ggml_build_forward_expand(graph, lidar); ggml_build_forward_expand(graph, dense_heatmap);
    if (!ggml_gallocr_alloc_graph(be.galloc, graph)) {
        error = "TransFusion heatmap graph allocation failed"; ggml_free(context); return false;
    }
    }
    if (!hm_reuse)
        ggml_backend_tensor_set(input, spatial.data(), 0, spatial.size() * sizeof(float));
    if (ggml_backend_graph_compute(be.be, graph) != GGML_STATUS_SUCCESS) {
        error = "TransFusion heatmap graph compute failed"; ggml_free(context); return false;
    }
    lap("head_graph_compute");
    std::vector<float> lidar_values = readback(lidar);
    std::vector<float> heatmap_values = readback(dense_heatmap);
    // bev_values (33 MB, the backbone_2d output) is consumed ONLY by the
    // debug tap below — defer its readback there so the production path
    // never pays it (measured: the three readbacks cost 44.9 ms vs 29 ms
    // for the whole graph compute).
    std::vector<float> bev_values;
    lap("head_readback");
    const int feature_width = static_cast<int>(lidar->ne[0]);
    const int feature_height = static_cast<int>(lidar->ne[1]);
    const int locations = feature_width * feature_height;
    if (sink) {
        bev_values = readback(bev);
        // readback gives GGML column-major [W, H, C]; reorder to [B*H*W, C] row-major HWC.
        int HW = feature_width * feature_height;
        int C_lidar = hp.transfusion_hidden_channels;
        std::vector<float> lidar_hwc((size_t) HW * C_lidar);
        for (int h = 0; h < feature_height; h++)
            for (int w = 0; w < feature_width; w++)
                for (int c = 0; c < C_lidar; c++) {
                    size_t src = (size_t) w + (size_t) feature_width * h +
                                 (size_t) feature_width * feature_height * c;
                    size_t dst = ((size_t) h * feature_width + w) * C_lidar + c;
                    lidar_hwc[dst] = lidar_values[src];
                }
        sink("transfusion.lidar", lidar_hwc.data(), HW, C_lidar);

        int C_hm = hp.num_class;
        std::vector<float> hm_hwc((size_t) HW * C_hm);
        for (int h = 0; h < feature_height; h++)
            for (int w = 0; w < feature_width; w++)
                for (int c = 0; c < C_hm; c++) {
                    size_t src = (size_t) w + (size_t) feature_width * h +
                                 (size_t) feature_width * feature_height * c;
                    size_t dst = ((size_t) h * feature_width + w) * C_hm + c;
                    hm_hwc[dst] = heatmap_values[src];
                }
        sink("transfusion.dense_heatmap", hm_hwc.data(), HW, C_hm);

        int C_backbone = static_cast<int>(bev->ne[2]);
        std::vector<float> bev_hwc((size_t) HW * C_backbone);
        for (int h = 0; h < feature_height; h++)
            for (int w = 0; w < feature_width; w++)
                for (int c = 0; c < C_backbone; c++) {
                    size_t src = (size_t) w + (size_t) feature_width * h +
                                 (size_t) feature_width * feature_height * c;
                    size_t dst = ((size_t) h * feature_width + w) * C_backbone + c;
                    bev_hwc[dst] = bev_values[src];
                }
        sink("bevfusion.backbone_2d", bev_hwc.data(), HW, C_backbone);
    }
    ggml_free(context);

    std::vector<float> initial_scores(heatmap_values.size());
    for (size_t i = 0; i < heatmap_values.size(); ++i)
        initial_scores[i] = 1.0f / (1.0f + std::exp(-heatmap_values[i]));
    std::vector<float> local_scores(initial_scores.size(), 0.0f);
    const int padding = hp.transfusion_nms_kernel_size / 2;
    for (int label = 0; label < hp.num_class; ++label)
        for (int y = 0; y < feature_height; ++y)
            for (int x = 0; x < feature_width; ++x) {
                const size_t index = static_cast<size_t>(x) + feature_width *
                    (y + static_cast<size_t>(feature_height) * label);
                float maximum = initial_scores[index];
                if (label < 8) {
                    if (x < padding || y < padding || x >= feature_width - padding ||
                        y >= feature_height - padding) continue;
                    for (int dy = -padding; dy <= padding; ++dy)
                        for (int dx = -padding; dx <= padding; ++dx)
                            maximum = std::max(maximum, initial_scores[
                                static_cast<size_t>(x + dx) + feature_width *
                                (y + dy + static_cast<size_t>(feature_height) * label)]);
                }
                if (initial_scores[index] == maximum) local_scores[index] = initial_scores[index];
            }
    std::vector<int32_t> order(local_scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + hp.transfusion_num_proposals, order.end(),
        [&](int32_t left, int32_t right) {
            if (local_scores[left] != local_scores[right]) return local_scores[left] > local_scores[right];
            return left < right;
        });
    order.resize(hp.transfusion_num_proposals);
    lap("proposal_select");

    const int proposals = hp.transfusion_num_proposals;
    const int channels = hp.transfusion_hidden_channels;
    std::vector<float> lidar_flat(static_cast<size_t>(channels) * locations);
    for (int location = 0; location < locations; ++location)
        for (int channel = 0; channel < channels; ++channel)
            lidar_flat[channel + static_cast<size_t>(channels) * location] =
                lidar_values[location + static_cast<size_t>(locations) * channel];
    std::vector<float> query(static_cast<size_t>(channels) * proposals);
    std::vector<float> category(static_cast<size_t>(hp.num_class) * proposals, 0.0f);
    std::vector<float> query_position(static_cast<size_t>(2) * proposals);
    std::vector<float> bev_position(static_cast<size_t>(2) * locations);
    std::vector<int32_t> labels(proposals), location_indices(proposals);
    std::vector<float> query_heatmap(static_cast<size_t>(hp.num_class) * proposals);
    for (int location = 0; location < locations; ++location) {
        bev_position[location * 2] = static_cast<float>(location % feature_width) + 0.5f;
        bev_position[location * 2 + 1] = static_cast<float>(location / feature_width) + 0.5f;
    }
    for (int proposal = 0; proposal < proposals; ++proposal) {
        labels[proposal] = order[proposal] / locations;
        location_indices[proposal] = order[proposal] % locations;
        category[labels[proposal] + static_cast<size_t>(hp.num_class) * proposal] = 1.0f;
        for (int channel = 0; channel < channels; ++channel)
            query[channel + static_cast<size_t>(channels) * proposal] =
                lidar_flat[channel + static_cast<size_t>(channels) * location_indices[proposal]];
        query_position[proposal * 2] = bev_position[location_indices[proposal] * 2];
        query_position[proposal * 2 + 1] = bev_position[location_indices[proposal] * 2 + 1];
        for (int label = 0; label < hp.num_class; ++label)
            query_heatmap[label + static_cast<size_t>(hp.num_class) * proposal] =
                local_scores[location_indices[proposal] + static_cast<size_t>(locations) * label];
    }

    if (sink) {
        // Proposal selection diagnostics: selected class and BEV location per
        // proposal (row-major, x + width*y), plus the masked dense score.
        std::vector<float> sel_labels(proposals), sel_index(proposals),
            sel_score(proposals);
        for (int p = 0; p < proposals; ++p) {
            sel_labels[p] = static_cast<float>(labels[p]);
            sel_index[p] = static_cast<float>(location_indices[p]);
            sel_score[p] = query_heatmap[labels[p] +
                static_cast<size_t>(hp.num_class) * p];
        }
        sink("transfusion.proposal_labels", sel_labels.data(), proposals, 1);
        sink("transfusion.proposal_index", sel_index.data(), proposals, 1);
        sink("transfusion.proposal_score", sel_score.data(), proposals, 1);
    }

    context = ggml_init(params);
    if (!context) { error = "TransFusion decoder context init failed"; return false; }
    ggml_tensor * lidar_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels, locations);
    ggml_tensor * query_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, channels, proposals);
    ggml_tensor * category_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, hp.num_class, proposals);
    ggml_tensor * query_pos_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 2, proposals);
    ggml_tensor * bev_pos_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 2, locations);
    ggml_set_input(lidar_input); ggml_set_input(query_input); ggml_set_input(category_input);
    ggml_set_input(query_pos_input); ggml_set_input(bev_pos_input);
    ggml_tensor * query_features = ggml_add(
        context, query_input, linear(context, weights.class_weight, weights.class_bias, category_input));
    ggml_tensor * query_initial = query_features;
    ggml_tensor * query_pos_embed = position_embedding(
        context, query_pos_input, weights.self_pos_weight1, weights.self_pos_bias1,
        weights.self_pos_scale, weights.self_pos_shift,
        weights.self_pos_weight2, weights.self_pos_bias2);
    ggml_tensor * bev_pos_embed = position_embedding(
        context, bev_pos_input, weights.cross_pos_weight1, weights.cross_pos_bias1,
        weights.cross_pos_scale, weights.cross_pos_shift,
        weights.cross_pos_weight2, weights.cross_pos_bias2);
    ggml_tensor * with_query_pos = ggml_add(context, query_features, query_pos_embed);
    ggml_tensor * self_result = attention(
        context, weights.self_q_weight, weights.self_q_bias,
        weights.self_k_weight, weights.self_k_bias,
        weights.self_v_weight, weights.self_v_bias,
        weights.self_out_weight, weights.self_out_bias,
        with_query_pos, with_query_pos, hp.transfusion_num_heads);
    query_features = affine_norm(context, ggml_add(context, query_features, self_result),
                                 weights.norm1_weight, weights.norm1_bias);
    ggml_tensor * query_after_self = query_features;
    with_query_pos = ggml_add(context, query_features, query_pos_embed);
    ggml_tensor * with_bev_pos = ggml_add(context, lidar_input, bev_pos_embed);
    ggml_tensor * cross_probabilities = nullptr;
    ggml_tensor * cross_merged = nullptr;
    ggml_tensor * cross_query = nullptr;
    ggml_tensor * cross_key = nullptr;
    ggml_tensor * cross_value = nullptr;
    ggml_tensor * cross_result = attention(
        context, weights.cross_q_weight, weights.cross_q_bias,
        weights.cross_k_weight, weights.cross_k_bias,
        weights.cross_v_weight, weights.cross_v_bias,
        weights.cross_out_weight, weights.cross_out_bias,
        with_query_pos, with_bev_pos, hp.transfusion_num_heads,
        &cross_probabilities, &cross_merged, &cross_query, &cross_key, &cross_value);
    query_features = affine_norm(context, ggml_add(context, query_features, cross_result),
                                 weights.norm2_weight, weights.norm2_bias);
    ggml_tensor * query_after_cross = query_features;
    ggml_tensor * feed_forward = linear(context, weights.linear1_weight,
                                        weights.linear1_bias, query_features);
    feed_forward = ggml_relu(context, feed_forward);
    feed_forward = linear(context, weights.linear2_weight, weights.linear2_bias, feed_forward);
    query_features = affine_norm(context, ggml_add(context, query_features, feed_forward),
                                 weights.norm3_weight, weights.norm3_bias);
    if (sink) {
        for (ggml_tensor * tap : {query_initial, query_pos_embed, self_result, query_after_self,
                                  bev_pos_embed, cross_probabilities, cross_merged,
                                  cross_query, cross_key, cross_value,
                                  cross_result, query_after_cross, query_features})
            ggml_set_output(tap);
    }
    ggml_tensor * center = prediction(context, weights.center, query_features);
    center = ggml_add(context, center, query_pos_input);
    ggml_tensor * center_z = prediction(context, weights.height, query_features);
    ggml_tensor * dimensions = prediction(context, weights.dimensions, query_features);
    ggml_tensor * rotation = prediction(context, weights.rotation, query_features);
    ggml_tensor * velocity = prediction(context, weights.velocity, query_features);
    ggml_tensor * refined_heatmap = prediction(context, weights.heatmap, query_features);
    for (ggml_tensor * output : {center, center_z, dimensions, rotation, velocity, refined_heatmap})
        ggml_set_output(output);
    graph = ggml_new_graph_custom(context, graph_nodes, false);
    for (ggml_tensor * output : {center, center_z, dimensions, rotation, velocity, refined_heatmap})
        ggml_build_forward_expand(graph, output);
    if (!ggml_gallocr_alloc_graph(be.galloc, graph)) {
        error = "TransFusion decoder graph allocation failed"; ggml_free(context); return false;
    }
    ggml_backend_tensor_set(lidar_input, lidar_flat.data(), 0, lidar_flat.size() * sizeof(float));
    ggml_backend_tensor_set(query_input, query.data(), 0, query.size() * sizeof(float));
    ggml_backend_tensor_set(category_input, category.data(), 0, category.size() * sizeof(float));
    ggml_backend_tensor_set(query_pos_input, query_position.data(), 0, query_position.size() * sizeof(float));
    ggml_backend_tensor_set(bev_pos_input, bev_position.data(), 0, bev_position.size() * sizeof(float));
    if (ggml_backend_graph_compute(be.be, graph) != GGML_STATUS_SUCCESS) {
        error = "TransFusion decoder graph compute failed"; ggml_free(context); return false;
    }
    if (sink) {
        const std::pair<const char *, ggml_tensor *> taps[] = {
            {"transfusion.query_initial", query_initial},
            {"transfusion.query_pos", query_pos_embed},
            {"transfusion.self_attention", self_result},
            {"transfusion.query_after_self", query_after_self},
            {"transfusion.bev_pos", bev_pos_embed},
            {"transfusion.cross_attention", cross_result},
            {"transfusion.cross_merged", cross_merged},
            {"transfusion.cross_query", cross_query},
            {"transfusion.cross_key", cross_key},
            {"transfusion.cross_value", cross_value},
            {"transfusion.query_after_cross", query_after_cross},
            {"transfusion.decoder", query_features},
        };
        for (const auto & tap : taps) {
            const std::vector<float> values = readback(tap.second);
            sink(tap.first, values.data(), tap.second->ne[1], tap.second->ne[0]);
        }
        std::vector<float> first_probabilities(static_cast<size_t>(locations));
        ggml_backend_tensor_get(cross_probabilities, first_probabilities.data(), 0,
                                first_probabilities.size() * sizeof(float));
        const float probability_sum = std::accumulate(
            first_probabilities.begin(), first_probabilities.end(), 0.0f);
        sink("transfusion.cross_probability_sum", &probability_sum, 1, 1);
        sink("transfusion.cross_probability_first", first_probabilities.data(), locations, 1);
    }
    lap("decoder_graph_readback");
    const std::vector<float> center_values = readback(center);
    const std::vector<float> height_values = readback(center_z);
    const std::vector<float> dimension_values = readback(dimensions);
    const std::vector<float> rotation_values = readback(rotation);
    const std::vector<float> velocity_values = readback(velocity);
    const std::vector<float> refined_values = readback(refined_heatmap);
    if (sink) sink("transfusion.query", refined_values.data(), proposals, hp.num_class);
    const auto & range = hp.transfusion_post_center_range;
    for (int proposal = 0; proposal < proposals; ++proposal) {
        const int label = labels[proposal];
        const float score = 1.0f / (1.0f + std::exp(
            -refined_values[label + static_cast<size_t>(hp.num_class) * proposal])) *
            query_heatmap[label + static_cast<size_t>(hp.num_class) * proposal];
        detection value{};
        value.x = center_values[proposal * 2] * hp.transfusion_feature_map_stride * hp.voxel_size[0] + hp.point_cloud_range[0];
        value.y = center_values[proposal * 2 + 1] * hp.transfusion_feature_map_stride * hp.voxel_size[1] + hp.point_cloud_range[1];
        value.z = height_values[proposal];
        value.dx = std::exp(dimension_values[proposal * 3]);
        value.dy = std::exp(dimension_values[proposal * 3 + 1]);
        value.dz = std::exp(dimension_values[proposal * 3 + 2]);
        value.heading = std::atan2(rotation_values[proposal * 2], rotation_values[proposal * 2 + 1]);
        value.score = score; value.label = label + 1;
        if (score > hp.score_thresh && value.x >= range[0] && value.y >= range[1] &&
            value.z >= range[2] && value.x <= range[3] && value.y <= range[4] && value.z <= range[5])
            detections.push_back(value);
    }
    std::sort(detections.begin(), detections.end(),
              [](const detection & left, const detection & right) { return left.score > right.score; });
    lap("decode_boxes");
    if (!hm_dump && tf_cache_on) {
        if (!g_tf_hm_cache.valid) {
            g_tf_hm_cache.valid = true;
            g_tf_hm_cache.context = context;
            g_tf_hm_cache.graph = graph;
            g_tf_hm_cache.input = input;
            g_tf_hm_cache.lidar = lidar;
        g_tf_hm_cache.bev = bev;
            g_tf_hm_cache.dense_heatmap = dense_heatmap;
            g_tf_hm_cache.W = width; g_tf_hm_cache.H = height;
            g_tf_hm_cache.C = hp.num_bev_features;
        }
    } else {
        ggml_free(context);
    }
    return true;
}

} // namespace openpcdet
