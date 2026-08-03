#include "antispoof_graph.hpp"
#include "align.hpp"
#include "backend.hpp"
#include "detect.hpp"
#include "graph_ops.hpp"
#include "preprocess.hpp"
#include "model_loader.hpp"
#include "model.hpp"
#include "common.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fd {

namespace {

// ONNX BatchNormalization epsilon (the MiniFASNet exports carry the default 1e-5).
constexpr float kAsBnEps = 1e-5f;

// Split `s` on `sep`, keeping empty fields (so a trailing empty attr field is
// preserved). Used to parse the embedded node spec "op;out;in_csv;attr_csv".
std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t pos = s.find(sep, start);
        if (pos == std::string::npos) { out.push_back(s.substr(start)); break; }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

// One parsed node: op type, output name, input names, and the Conv attrs
// (stride/pad/group, defaulted when absent).
struct Node {
    std::string op;
    std::string out;
    std::vector<std::string> in;
    int stride = 1, pad = 0, group = 1;
    int transB = 0;   // ONNX Gemm: 1 when B is stored pre-transposed [out,in]
    int kernel = 0;   // pooling kernel size (MaxPool / AveragePool)
    int rc = 0;       // Reshape last dim: reshape to [-1, rc]
    float eps = -1.0f;  // per-node BatchNormalization epsilon (<0 -> use bn_eps)
    std::vector<int> perm;  // Transpose ONNX perm (4-D heads use [2,3,0,1])
};

// A genuine grouped convolution (group > 1 with more than one input channel per
// group, e.g. MobileFaceNet's group=64 / 2-in-2-out layer) that is neither dense
// nor depthwise. ggml has no native grouped conv, so split the input channels and
// output filters into `group` equal slices, run a dense conv per slice, and concat
// the results back along the channel axis. W ne = [KW,KH,ipg,OC]; x ne = [W,H,IC,1]
// with IC = group*ipg, OC = group*opg.
ggml_tensor* conv_grouped(ggml_context* ctx, ggml_tensor* W, ggml_tensor* x,
                          int group, int stride, int pad) {
    const int64_t ipg = W->ne[2];                 // input channels per group
    const int64_t opg = W->ne[3] / group;         // output filters per group
    ggml_tensor* acc = nullptr;
    for (int g = 0; g < group; ++g) {
        ggml_tensor* xg = ggml_cont(ctx, ggml_view_4d(
            ctx, x, x->ne[0], x->ne[1], ipg, 1,
            x->nb[1], x->nb[2], x->nb[3], (size_t)g * ipg * x->nb[2]));
        ggml_tensor* wg = ggml_cont(ctx, ggml_view_4d(
            ctx, W, W->ne[0], W->ne[1], ipg, opg,
            W->nb[1], W->nb[2], W->nb[3], (size_t)g * opg * W->nb[3]));
        ggml_tensor* yg = ggml_conv_2d(ctx, wg, xg, stride, stride, pad, pad, 1, 1);
        acc = acc ? ggml_concat(ctx, acc, yg, 2) : yg;
    }
    return acc;
}

// One folded Conv+BatchNorm: BN's inference affine (scale = gamma/sqrt(var+eps),
// shift = beta - mean*scale) folded into the PRECEDING conv's weights and bias,
// so the BatchNormalization op vanishes from the graph (no per-output full-map
// mul+add - the ~20% of the SFace recognizer forward that the unfused BN cost).
// W'[oc] = W[oc]*scale[oc]; b'[oc] = b[oc]*scale[oc] + shift[oc] (b=0 for the
// bias-less SFace/MobileFaceNet convs). Algebraically identical to the live-BN
// path; the only delta is f32 rounding order ((w*s)*x vs (sum w*x)*s), far inside
// the recognizer max|d|<=1e-3 / cosine>=0.9999 gate. F32 conv weights only (the
// host fold dequantizes nothing; quantized/f16 packs keep the live-BN path).
struct ConvBNFold {
    std::vector<float> w;     // folded weights, original conv-weight layout/ne
    std::vector<float> bias;  // folded per-output-channel bias
    int64_t ne[4] = {0, 0, 0, 0};
};

// Process-lifetime cache keyed by the (stable, unique-per-model) conv weight
// tensor pointer. Node-based map storage keeps the float buffers' addresses
// stable, so the host pointers handed to graph_input_tensor safely outlive the
// deferred host->device input copy of any later compute.
std::unordered_map<const ggml_tensor*, ConvBNFold> g_convbn_fold;
std::mutex g_convbn_fold_mu;

const ConvBNFold& convbn_fold(const ModelLoader& ml, const std::string& prefix,
                              const std::string& w_name, const std::string& bias_name,
                              const std::string& gamma_name, const std::string& beta_name,
                              const std::string& mean_name, const std::string& var_name,
                              float eps) {
    ggml_tensor* W = clone_weight(nullptr, ml, (prefix + w_name).c_str());
    {
        std::lock_guard<std::mutex> lk(g_convbn_fold_mu);
        auto it = g_convbn_fold.find(W);
        if (it != g_convbn_fold.end()) return it->second;
    }
    ggml_tensor* gamma = clone_weight(nullptr, ml, (prefix + gamma_name).c_str());
    ggml_tensor* beta  = clone_weight(nullptr, ml, (prefix + beta_name).c_str());
    ggml_tensor* mean  = clone_weight(nullptr, ml, (prefix + mean_name).c_str());
    ggml_tensor* var   = clone_weight(nullptr, ml, (prefix + var_name).c_str());
    const int64_t OC = W->ne[3];
    const int64_t inner = W->ne[0] * W->ne[1] * W->ne[2];
    std::vector<float> g(OC), b(OC), m(OC), v(OC), wsrc((size_t)OC * inner);
    ggml_backend_tensor_get(gamma, g.data(), 0, OC * sizeof(float));
    ggml_backend_tensor_get(beta,  b.data(), 0, OC * sizeof(float));
    ggml_backend_tensor_get(mean,  m.data(), 0, OC * sizeof(float));
    ggml_backend_tensor_get(var,   v.data(), 0, OC * sizeof(float));
    ggml_backend_tensor_get(W, wsrc.data(), 0, (size_t)OC * inner * sizeof(float));
    std::vector<float> bsrc;
    if (!bias_name.empty() && ml.tensor((prefix + bias_name).c_str())) {
        bsrc.resize(OC);
        ggml_tensor* B = clone_weight(nullptr, ml, (prefix + bias_name).c_str());
        ggml_backend_tensor_get(B, bsrc.data(), 0, OC * sizeof(float));
    }
    ConvBNFold f;
    f.ne[0] = W->ne[0]; f.ne[1] = W->ne[1]; f.ne[2] = W->ne[2]; f.ne[3] = W->ne[3];
    f.w.resize((size_t)OC * inner);
    f.bias.resize(OC);
    for (int64_t oc = 0; oc < OC; ++oc) {
        const float s  = g[oc] / std::sqrt(v[oc] + eps);
        const float sh = b[oc] - m[oc] * s;
        const float* wr = wsrc.data() + (size_t)oc * inner;
        float*       wo = f.w.data()  + (size_t)oc * inner;
        for (int64_t i = 0; i < inner; ++i) wo[i] = wr[i] * s;
        f.bias[oc] = (bsrc.empty() ? 0.0f : bsrc[oc] * s) + sh;
    }
    std::lock_guard<std::mutex> lk(g_convbn_fold_mu);
    return g_convbn_fold.emplace(W, std::move(f)).first->second;
}

// Parse one embedded node spec. `insep` separates the input list and the attr
// list: the MiniFASNet / MobileFaceNet graphs use ',', the SCRFD detector graph
// uses '|' (its head weight names embed commas, e.g. "(8, 8)").
Node parse_node(const std::string& spec, char insep) {
    std::vector<std::string> f = split(spec, ';');
    Node n;
    n.op = f.size() > 0 ? f[0] : "";
    n.out = f.size() > 1 ? f[1] : "";
    if (f.size() > 2 && !f[2].empty()) n.in = split(f[2], insep);
    if (f.size() > 3 && !f[3].empty()) {
        for (const std::string& kv : split(f[3], insep)) {
            std::vector<std::string> p = split(kv, '=');
            if (p.size() != 2) continue;
            if (p[0] == "pm") {                       // Transpose perm "a/b/c/d"
                for (const std::string& a : split(p[1], '/'))
                    n.perm.push_back(std::atoi(a.c_str()));
                continue;
            }
            if (p[0] == "e") {                        // BN epsilon (float, e.g. 2e-05)
                n.eps = (float)std::atof(p[1].c_str());
                continue;
            }
            const int v = std::atoi(p[1].c_str());
            if (p[0] == "s") n.stride = v;
            else if (p[0] == "p") n.pad = v;
            else if (p[0] == "g") n.group = v;
            else if (p[0] == "tb") n.transB = v;
            else if (p[0] == "k") n.kernel = v;
            else if (p[0] == "rc") n.rc = v;
        }
    }
    return n;
}

// Map an ONNX `Transpose` perm onto a ggml permute. ggml's axis order is the
// reverse of ONNX's (ggml ne[0] is ONNX's innermost), so for a 4-D tensor ONNX
// axis `a` is ggml axis `3-a`. ONNX out axis i takes in axis perm[i]; we want the
// ggml source axis (3-perm[i]) to land at ggml dest (3-i). Returns a permuted
// VIEW (callers ggml_cont it, e.g. the following Reshape).
ggml_tensor* transpose_onnx(ggml_context* ctx, ggml_tensor* x,
                            const std::vector<int>& perm) {
    int dst[4] = {0, 1, 2, 3};
    for (int i = 0; i < 4; ++i) dst[3 - perm[i]] = 3 - i;
    return ggml_permute(ctx, x, dst[0], dst[1], dst[2], dst[3]);
}

// Execute ONE parsed node, resolving inputs against `vals` (prior activations) or
// the `prefix`-namespaced GGUF weights. Shared by the single-output runner (the
// MiniFASNet / MobileFaceNet graphs) and the multi-output SCRFD detector runner;
// `keep` owns BN-eps host scalars that must outlive the compute.
ggml_tensor* build_node(ggml_context* ctx, const ModelLoader& ml,
                        const std::string& prefix, const Node& n,
                        std::unordered_map<std::string, ggml_tensor*>& vals,
                        std::vector<std::vector<float>>& keep, float bn_eps,
                        const std::unordered_map<std::string, Node>* folds,
                        const std::unordered_set<std::string>* fuse_convs,
                        std::unordered_map<std::string, ggml_tensor*>* pending_bias) {
    auto resolve = [&](const std::string& nm) -> ggml_tensor* {
        auto it = vals.find(nm);
        if (it != vals.end()) return it->second;
        return clone_weight(ctx, ml, (prefix + nm).c_str());
    };
    if (n.op == "Conv") {
        ggml_tensor* x = resolve(n.in[0]);
        ggml_tensor* W = nullptr;     // conv kernel (raw weight, or folded leaf)
        ggml_tensor* B = nullptr;     // bias (raw weight, folded leaf, or none)
        // Conv+BN fold: if this conv's output is a recorded fold target AND its
        // weight is F32, replace W with the host-folded W*scale and add the folded
        // per-channel bias; the matched BatchNormalization node aliases this result
        // (see run_onnx_graph), so the BN op never enters the graph.
        ggml_tensor* Wraw = clone_weight(ctx, ml, (prefix + n.in[1]).c_str());
        const Node* bn = nullptr;
        if (folds) { auto it = folds->find(n.out); if (it != folds->end()) bn = &it->second; }
        if (bn && Wraw->type == GGML_TYPE_F32) {
            const std::string biasname = (n.in.size() > 2) ? n.in[2] : std::string();
            const ConvBNFold& f = convbn_fold(ml, prefix, n.in[1], biasname,
                                              bn->in[1], bn->in[2], bn->in[3], bn->in[4],
                                              bn->eps >= 0.0f ? bn->eps : bn_eps);
            W = graph_input_tensor(ctx, GGML_TYPE_F32, 4, f.ne,
                                   f.w.data(), f.w.size() * sizeof(float));
            const int64_t bne[4] = {(int64_t)f.bias.size(), 1, 1, 1};
            B = graph_input_tensor(ctx, GGML_TYPE_F32, 1, bne,
                                   f.bias.data(), f.bias.size() * sizeof(float));
        } else {
            W = Wraw;
            if (n.in.size() > 2 && !n.in[2].empty())
                B = clone_weight(ctx, ml, (prefix + n.in[2]).c_str());
        }
        if (n.group == 1)
            x = ggml_conv_2d(ctx, W, x, n.stride, n.stride, n.pad, n.pad, 1, 1);
        else if (W->ne[2] == 1)  // depthwise (1 in-channel/group): ne [KW,KH,1,C]
            x = ggml_conv_2d_dw_direct(ctx, W, x, n.stride, n.stride,
                                       n.pad, n.pad, 1, 1);
        else  // genuine grouped conv (>1 in-channel/group): split + concat
            x = conv_grouped(ctx, W, x, n.group, n.stride, n.pad);
        // When the following BN->PReLU chain will be fused (CPU), defer the bias:
        // the fused_prelu_bias op adds it in the same pass as the activation, so
        // skip the standalone broadcast add here and hand the leaf to the PReLU.
        const bool defer_bias = B && pending_bias && fuse_convs &&
                                fuse_convs->count(n.out) != 0;
        if (defer_bias) {
            (*pending_bias)[n.out] = B;
        } else if (B) {
            const int64_t OC = x->ne[2];
            x = ggml_add(ctx, x, ggml_reshape_4d(ctx, B, 1, 1, OC, 1));
        }
        return x;
    } else if (n.op == "PRelu") {
        ggml_tensor* x     = resolve(n.in[0]);
        ggml_tensor* alpha = clone_weight(ctx, ml, (prefix + n.in[1]).c_str());
        if (pending_bias) {
            auto it = pending_bias->find(n.in[0]);
            if (it != pending_bias->end())   // fused (deferred-bias + PReLU) pass
                return fused_prelu_bias(ctx, x, it->second, alpha);
        }
        return prelu(ctx, x, alpha);
    } else if (n.op == "Relu") {
        return ggml_relu(ctx, resolve(n.in[0]));
    } else if (n.op == "Sigmoid") {
        return ggml_sigmoid(ctx, resolve(n.in[0]));
    } else if (n.op == "Add") {
        return ggml_add(ctx, resolve(n.in[0]), resolve(n.in[1]));
    } else if (n.op == "Sub") {
        // SFace in-graph input normalization: data - scalar_op1 (mean 127.5). The
        // subtrahend is a [1] leaf that broadcasts over the whole [W,H,C,1] input.
        return ggml_sub(ctx, resolve(n.in[0]), resolve(n.in[1]));
    } else if (n.op == "Dropout") {
        // Inference-time Dropout is the identity (the ratio is a no-op at eval).
        return resolve(n.in[0]);
    } else if (n.op == "Identity") {
        // 1k3d68 begins with an Identity passthrough (data -> id) before bn_data.
        return resolve(n.in[0]);
    } else if (n.op == "Mul") {
        // SE scale: x [W,H,C,1] * gate [1,1,C,1]; or the SCRFD GFL per-stride
        // bbox scale: reg [W,H,C,1] * scalar [1] (det_2.5g). Both broadcast.
        return ggml_mul(ctx, resolve(n.in[0]), resolve(n.in[1]));
    } else if (n.op == "GlobalAveragePool") {
        ggml_tensor* x = resolve(n.in[0]);
        const int W = (int)x->ne[0], H = (int)x->ne[1];
        return ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, W, H, W, H, 0, 0);
    } else if (n.op == "MaxPool" || n.op == "AveragePool") {
        // SCRFD stem MaxPool / residual-shortcut AveragePool. The square 640 input
        // keeps every pooled map even, so ceil_mode is moot (ceil == floor).
        ggml_tensor* x = resolve(n.in[0]);
        const ggml_op_pool op = (n.op == "MaxPool") ? GGML_OP_POOL_MAX
                                                    : GGML_OP_POOL_AVG;
        const int k = n.kernel > 0 ? n.kernel : 2;
        const int s = n.stride > 0 ? n.stride : k;
        return ggml_pool_2d(ctx, x, op, k, k, s, s, (float)n.pad, (float)n.pad);
    } else if (n.op == "Resize") {
        // SCRFD FPN top-down upsample: nearest, 2x (the shape-plumbing that fed the
        // dynamic target size was dropped by the converter; on the even 640 maps it
        // is exactly 2x, matching the det_10g hand path's ggml_upscale).
        return ggml_upscale(ctx, resolve(n.in[0]), 2, GGML_SCALE_MODE_NEAREST);
    } else if (n.op == "Transpose") {
        return transpose_onnx(ctx, resolve(n.in[0]), n.perm);
    } else if (n.op == "Reshape") {
        // SCRFD head reshape to [-1, rc]: ggml ne [rc, nelem/rc], contiguous. After
        // the [2,3,0,1] Transpose the contiguous element order is ONNX row-major
        // (h,w,anchor,c), so this lands the (H*W*anchor, c) layout the host decode
        // reads (== flatten_head), with rc = 1 (score) / 4 (bbox) / 10 (kps).
        ggml_tensor* x = ggml_cont(ctx, resolve(n.in[0]));
        const int64_t C = n.rc > 0 ? n.rc : 1;
        return ggml_reshape_2d(ctx, x, C, ggml_nelements(x) / C);
    } else if (n.op == "Flatten") {
        ggml_tensor* x = ggml_cont(ctx, resolve(n.in[0]));
        const int64_t N = x->ne[3];
        return ggml_reshape_2d(ctx, x, ggml_nelements(x) / N, N);
    } else if (n.op == "MatMul") {
        // ONNX weight is [in,out] -> ggml ne [out,in]; transpose so the
        // contraction dim (in) is ggml ne[0] for ggml_mul_mat.
        ggml_tensor* x = resolve(n.in[0]);
        ggml_tensor* W = clone_weight(ctx, ml, (prefix + n.in[1]).c_str());
        return ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, W)), x);
    } else if (n.op == "Gemm") {
        // y = alpha*A.B(^T) + beta*C, with the ArcFace head's alpha=beta=1.
        // MobileFaceNet's `fc` Gemm has transB=1, so B is stored [out,in] ->
        // ggml ne [in,out], whose ne[0] (in) is already the contraction dim
        // for ggml_mul_mat (no transpose, unlike the transB=0 MatMul above).
        ggml_tensor* x = resolve(n.in[0]);
        ggml_tensor* W = clone_weight(ctx, ml, (prefix + n.in[1]).c_str());
        if (!n.transB) W = ggml_cont(ctx, ggml_transpose(ctx, W));
        ggml_tensor* y = ggml_mul_mat(ctx, W, x);
        if (n.in.size() > 2 && !n.in[2].empty())
            y = ggml_add(ctx, y, clone_weight(ctx, ml, (prefix + n.in[2]).c_str()));
        return y;
    } else if (n.op == "BatchNormalization") {
        // BN over the channel dim, for BOTH 4-D spatial maps ([W,H,C,N], the SFace
        // conv BNs) and 1-D linear outputs ([C,N], the MobileFaceNet / SFace fc1
        // BN). The per-node epsilon `n.eps` is used when present (SFace mixes 1e-3
        // conv BNs with 2e-5 bn1/fc1 BNs); otherwise the caller's `bn_eps`.
        ggml_tensor* x     = resolve(n.in[0]);
        ggml_tensor* gamma = clone_weight(ctx, ml, (prefix + n.in[1]).c_str());
        ggml_tensor* beta  = clone_weight(ctx, ml, (prefix + n.in[2]).c_str());
        ggml_tensor* mean  = clone_weight(ctx, ml, (prefix + n.in[3]).c_str());
        ggml_tensor* var   = clone_weight(ctx, ml, (prefix + n.in[4]).c_str());
        keep.emplace_back(1, n.eps >= 0.0f ? n.eps : bn_eps);
        const int64_t ne1[4] = {1, 1, 1, 1};
        ggml_tensor* eps = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne1,
                                              keep.back().data(), sizeof(float));
        ggml_tensor* inv   = ggml_sqrt(ctx, ggml_add1(ctx, var, eps));
        ggml_tensor* scale = ggml_div(ctx, gamma, inv);
        ggml_tensor* shift = ggml_sub(ctx, beta, ggml_mul(ctx, mean, scale));
        const int64_t C = gamma->ne[0];
        // Linear activation: channel is ggml ne[0] ([C,1]); broadcast over the row.
        if (x->ne[0] == C && x->ne[1] == 1 && x->ne[2] == 1) {
            return ggml_add(ctx, ggml_mul(ctx, x, ggml_reshape_2d(ctx, scale, C, 1)),
                            ggml_reshape_2d(ctx, shift, C, 1));
        }
        // Spatial map: channel is ggml ne[2]; broadcast scale/shift per channel.
        ggml_tensor* s4 = ggml_reshape_4d(ctx, scale, 1, 1, C, 1);
        ggml_tensor* b4 = ggml_reshape_4d(ctx, shift, 1, 1, C, 1);
        return ggml_add(ctx, ggml_mul(ctx, x, s4), b4);
    }
    throw std::runtime_error("onnx_graph: unsupported op " + n.op);
}

} // namespace

std::vector<float> run_onnx_graph(const ModelLoader& ml, const std::string& prefix,
                                  const std::vector<std::string>& specs,
                                  const std::string& out_name,
                                  const std::string& input_name,
                                  const std::vector<float>& blob, int size,
                                  Backend& be, float bn_eps) {
    if (specs.empty())
        throw std::runtime_error("onnx_graph: empty graph topology");

    std::vector<std::vector<float>> keep;  // BN-eps host buffers; outlive compute
    std::vector<float> out;
    const bool ok = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        std::unordered_map<std::string, ggml_tensor*> vals;

        const int64_t ne[4] = {size, size, 3, 1};
        ggml_tensor* input = graph_input_tensor(ctx, GGML_TYPE_F32, 4, ne,
                                                blob.data(), blob.size() * sizeof(float));
        // Seed the input under BOTH a generic alias and the graph's real ONNX
        // input name (e.g. MiniFASNet "input" vs MobileFaceNet "input.1"), so the
        // first node resolves it as an activation rather than a missing weight.
        vals["input"] = input;
        if (!input_name.empty()) vals[input_name] = input;

        // Parse once, then plan Conv+BN folds: a BatchNormalization whose in[0] is
        // produced by a Conv that feeds nothing else (single consumer) folds into
        // that conv (host weight*scale + bias). The matched BN is then a pure alias
        // of the folded conv output. On the SFace recognizer this elides all 29
        // spatial BNs; the SCRFD detector graph carries no BN nodes (folded at
        // conversion) and the MiniFASNet antispoof BN sits after a MatMul, so
        // neither matches and both keep their exact prior graph.
        std::vector<Node> nodes;
        nodes.reserve(specs.size());
        for (const std::string& spec : specs) nodes.push_back(parse_node(spec, ','));
        std::unordered_map<std::string, int> consumers;
        std::unordered_map<std::string, size_t> producer;
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (const std::string& in : nodes[i].in) consumers[in]++;
            producer[nodes[i].out] = i;
        }
        std::unordered_map<std::string, const Node*> consumer_node;
        for (const Node& n : nodes)
            for (const std::string& in : n.in) consumer_node[in] = &n;

        std::unordered_map<std::string, Node> folds;          // conv.out -> BN node
        std::unordered_map<std::string, std::string> bn_alias; // BN.out -> conv.out
        std::unordered_set<std::string> fuse_convs;            // conv.out, CPU fuse
        // A/B overrides (parity-neutral; both default ON where applicable):
        //   FACEDETECT_NO_BN_FOLD    - keep BN as live nodes (disables fold+fuse)
        //   FACEDETECT_NO_PRELU_FUSE - keep the separate bias-add + PReLU nodes
        const bool no_fold = std::getenv("FACEDETECT_NO_BN_FOLD") != nullptr;
        const bool no_fuse = std::getenv("FACEDETECT_NO_PRELU_FUSE") != nullptr;
        const bool cpu = backend_is_cpu();
        for (const Node& n : nodes) {
            if (no_fold) break;
            if (n.op != "BatchNormalization" || n.in.size() < 5) continue;
            auto pit = producer.find(n.in[0]);
            if (pit == producer.end()) continue;
            const Node& conv = nodes[pit->second];
            if (conv.op != "Conv" || consumers[n.in[0]] != 1) continue;
            folds[conv.out] = n;
            bn_alias[n.out] = conv.out;
            // Fuse the deferred bias-add into the following PReLU when (a) on CPU,
            // (b) the BN feeds exactly one consumer, and (c) that consumer is a
            // PReLU reading the BN output directly. The SFace/MobileFaceNet stack
            // is uniformly Conv->BN->PReLU, so every fold also fuses.
            if (cpu && !no_fuse && consumers[n.out] == 1) {
                auto cit = consumer_node.find(n.out);
                if (cit != consumer_node.end() && cit->second->op == "PRelu" &&
                    !cit->second->in.empty() && cit->second->in[0] == n.out)
                    fuse_convs.insert(conv.out);
            }
        }

        std::unordered_map<std::string, ggml_tensor*> pending_bias;
        ggml_tensor* last = nullptr;
        for (const Node& n : nodes) {
            auto ba = bn_alias.find(n.out);
            if (ba != bn_alias.end()) {           // folded BN: alias the conv output
                ggml_tensor* y = vals.at(ba->second);
                vals[n.out] = y;
                // carry the conv's deferred bias leaf onto the BN alias name, so
                // the PReLU (which reads the BN output) finds it.
                auto pb = pending_bias.find(ba->second);
                if (pb != pending_bias.end()) pending_bias[n.out] = pb->second;
                last = y;
                continue;
            }
            ggml_tensor* y = build_node(ctx, ml, prefix, n, vals, keep, bn_eps,
                                        &folds, &fuse_convs, &pending_bias);
            vals[n.out] = y;
            last = y;
        }
        auto it = vals.find(out_name);
        return it != vals.end() ? it->second : last;
    }, out);
    if (!ok) throw std::runtime_error("onnx_graph: compute failed");
    return out;
}

std::vector<std::vector<float>> run_onnx_graph_multi(
    const ModelLoader& ml, const std::string& prefix,
    const std::vector<std::string>& specs, const std::string& input_name,
    const std::vector<std::string>& output_names,
    const std::vector<float>& blob, int size, Backend& be, float bn_eps) {
    if (specs.empty())
        throw std::runtime_error("onnx_graph: empty graph topology");
    if (output_names.empty())
        throw std::runtime_error("onnx_graph: no graph outputs requested");

    std::vector<std::vector<float>> keep;  // BN-eps host buffers; outlive compute
    std::vector<std::vector<float>> outs(output_names.size());
    std::vector<float> dummy;
    const bool ok = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        std::unordered_map<std::string, ggml_tensor*> vals;

        const int64_t ne[4] = {size, size, 3, 1};
        ggml_tensor* input = graph_input_tensor(ctx, GGML_TYPE_F32, 4, ne,
                                                blob.data(), blob.size() * sizeof(float));
        vals["input"] = input;
        if (!input_name.empty()) vals[input_name] = input;

        for (const std::string& spec : specs) {
            Node n = parse_node(spec, '|');
            vals[n.out] = build_node(ctx, ml, prefix, n, vals, keep, bn_eps,
                                     nullptr, nullptr, nullptr);
        }
        // Capture every requested output (the SCRFD per-stride heads) for
        // host-side readback. Returns the last as the graph's nominal root.
        ggml_tensor* root = nullptr;
        for (size_t i = 0; i < output_names.size(); ++i) {
            auto it = vals.find(output_names[i]);
            if (it == vals.end())
                throw std::runtime_error("onnx_graph: missing output " + output_names[i]);
            capture_graph_output(it->second, &outs[i]);
            root = it->second;
        }
        return root;
    }, dummy);
    if (!ok) throw std::runtime_error("onnx_graph: compute failed");
    return outs;
}

std::vector<float> minifasnet_forward(const ModelLoader& ml, int model_idx,
                                      const std::vector<float>& blob, int size,
                                      Backend& be) {
    const FaceConfig& cfg = ml.config();
    if (model_idx < 0 || (size_t)model_idx >= cfg.antispoof_graphs.size())
        throw std::runtime_error("antispoof: model index out of range");
    const std::string prefix = "as" + std::to_string(model_idx) + ".";
    return run_onnx_graph(ml, prefix, cfg.antispoof_graphs[model_idx],
                          cfg.antispoof_graph_outputs[model_idx],
                          /*input_name=*/"input", blob, size, be, kAsBnEps);
}

bool antispoof_crop(const Image& img, const Detection& d, float scale, Image& out,
                    int size) {
    const float src_w = (float)img.width, src_h = (float)img.height;
    const float box_w = std::max(1.0f, d.x2 - d.x1);
    const float box_h = std::max(1.0f, d.y2 - d.y1);
    // Clamp the expansion so the crop fits inside the source image (reference
    // _crop_face: scale = min((H-1)/h, (W-1)/w, scale)).
    scale = std::min(std::min((src_h - 1.0f) / box_h, (src_w - 1.0f) / box_w), scale);
    const float new_w = box_w * scale, new_h = box_h * scale;
    const float cx = d.x1 + box_w * 0.5f, cy = d.y1 + box_h * 0.5f;
    // int() truncates toward zero, then clamp to the image (matches numpy/cv2).
    const int cx1 = std::max(0, (int)(cx - new_w * 0.5f));
    const int cy1 = std::max(0, (int)(cy - new_h * 0.5f));
    const int cx2 = std::min((int)src_w - 1, (int)(cx + new_w * 0.5f));
    const int cy2 = std::min((int)src_h - 1, (int)(cy + new_h * 0.5f));
    const float cw = (float)(cx2 - cx1 + 1), ch = (float)(cy2 - cy1 + 1);
    if (cw <= 0.0f || ch <= 0.0f) return false;
    // Express the integer crop + cv2.resize(INTER_LINEAR, align_corners=False) as
    // a forward affine src->dst and reuse the shared cv2-faithful warp:
    //   u = (src_x - cx1 + 0.5)*size/cw - 0.5.
    const float sx = (float)size / cw, sy = (float)size / ch;
    const std::array<float, 6> M{ sx, 0.0f, (0.5f - (float)cx1) * sx - 0.5f,
                                  0.0f, sy, (0.5f - (float)cy1) * sy - 0.5f };
    return warp_affine(img, M, out, size, size);
}

namespace {

// Averaged ensemble softmax over the 3 classes (reference Antispoofer.predict).
std::array<double, 3> ensemble_softmax(const Model& m, const Image& img,
                                       const Detection& d) {
    const FaceConfig& cfg = m.config();
    const int size = (int)cfg.antispoof_input_size;
    const std::vector<float>& sc = cfg.antispoof_scales;
    std::array<double, 3> accum{0.0, 0.0, 0.0};
    for (size_t i = 0; i < sc.size(); ++i) {
        Image crop;
        if (!antispoof_crop(img, d, sc[i], crop, size))
            throw std::runtime_error("antispoof: crop failed");
        // Raw BGR planes (swap_rb=true on the RGB Image), no mean/std: exactly
        // what the reference feeds onnxruntime.
        std::vector<float> blob = to_blob(crop, size, 0.0f, 1.0f, /*swap_rb=*/true);
        std::vector<float> logits = minifasnet_forward(m.loader(), (int)i, blob,
                                                       size, global_backend());
        FD_ASSERT(logits.size() == 3);
        const float mx = std::max({logits[0], logits[1], logits[2]});
        double e[3], s = 0.0;
        for (int k = 0; k < 3; ++k) { e[k] = std::exp((double)logits[k] - mx); s += e[k]; }
        for (int k = 0; k < 3; ++k) accum[k] += e[k] / s;
    }
    const double n = sc.empty() ? 1.0 : (double)sc.size();
    for (int k = 0; k < 3; ++k) accum[k] /= n;
    return accum;
}

} // namespace

float antispoof_real_prob(const Model& m, const Image& img, const Detection& d) {
    return (float)ensemble_softmax(m, img, d)[1];
}

float antispoof_score(const Model& m, const Image& img, const Detection& d) {
    const std::array<double, 3> a = ensemble_softmax(m, img, d);
    const int arg = (a[1] >= a[0] && a[1] >= a[2]) ? 1 : (a[0] >= a[2] ? 0 : 2);
    return (arg == 1) ? (float)a[1] : 0.0f;  // REAL_CLASS_IDX = 1
}

} // namespace fd
