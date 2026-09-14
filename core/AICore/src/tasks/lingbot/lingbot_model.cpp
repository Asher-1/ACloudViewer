// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "lingbot_model.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/ggml_backend_registry.hpp"
#include "lingbot_gguf_loader.h"

namespace lingbot {

// Backend acquisition through the AICore runtime lease layer (implementation
// detail of this translation unit). Physical ggml backends are shared
// process-wide through BackendLease; the upstream lingbot-map setenv-based
// GGML_VK_* precision switches are NOT carried over: the lingbot graph
// reaches the same exact-scalar contract through graph-node markers
// ("lingbot_scalar_*" name routing, see the lingbot_merged ggml patch)
// without any process-global state.
class backend {
public:
    // Resolves `name` ("auto", "cpu", "gpu", "vulkan[:N]", "cuda[:N]",
    // "CUDA0", ...) through the AICore runtime order (CUDA -> Vulkan -> CPU
    // on Linux/Windows, Metal -> CPU on macOS) and holds a lease on the
    // resolved physical backend for this context's lifetime.
    bool init(const std::string &name, int threads);
    void release();
    ggml_backend_t get() const { return backend_; }
    // Backend-RESOLVED device name ("CUDA0", "Vulkan0", "cpu", ...").
    const std::string &device_name() const { return device_name_; }
    const std::string &error() const { return error_; }

private:
    aicore::runtime::BackendLease lease_;
    ggml_backend_t backend_ = nullptr;
    std::string device_name_;
    std::string error_;
};

bool backend::init(const std::string &name, int threads) {
    release();
    // Normalize ggml-style device names ("CUDA0", "Vulkan0") into the AICore
    // device-request syntax ("cuda:0", "vulkan:0"); callers may pass either.
    std::string want = name.empty() ? std::string("auto") : name;
    for (char &c : want) c = static_cast<char>(std::tolower(c));
    if (want.rfind("cuda", 0) == 0 && want.size() > 4) {
        want = "cuda:" + want.substr(4);
    } else if (want.rfind("vulkan", 0) == 0 && want.size() > 6) {
        want = "vulkan:" + want.substr(6);
    }
    std::string lease_error;
    lease_ =
            aicore::runtime::acquire_backend_lease(want, threads, &lease_error);
    if (!lease_) {
        error_ = lease_error.empty()
                         ? ("requested ggml backend is unavailable: " + name)
                         : lease_error;
        return false;
    }
    backend_ = lease_.handle();
    device_name_ = lease_.device();
    return backend_ != nullptr;
}

void backend::release() {
    backend_ = nullptr;
    device_name_.clear();
    lease_.reset();
    error_.clear();
}

// Explicit inference options, set once by model::load from the caller's
// model_options and read throughout this translation unit. Replaces the
// former LINGBOT_* environment reads (see include/lingbot_map.h).
// thread_local (AICore adaptation): each model context configures it on its
// own worker thread before load/infer, so concurrent contexts on different
// threads never share this pointer (each public context owns its state).
static thread_local const model_options *g_opt = nullptr;

namespace {

struct graph_builder {
    struct kv_state {
        // Values are stored in GGML's contiguous [head_dim, heads, tokens]
        // order. Global attention keeps all frame-special tokens, but only a
        // bounded scale+recent patch window, matching the Python streaming
        // cache instead of growing O(sequence_length * patch_count).
        std::vector<float> k, v;
        std::vector<float> scale_k, scale_v;
        std::vector<float> live_k, live_v;
        std::vector<float> special_k, special_v;
        int64_t tokens = 0;
        int64_t frame_count = 0;
        int64_t token_width = 0;
        int64_t patch_tokens = 0;
    };
    struct kv_capture {
        std::string key;
        ggml_tensor *k = nullptr;
        ggml_tensor *v = nullptr;
        int64_t tokens = 0;
        int64_t head_dim = 0;
        int64_t heads = 0;
        // A scale pass is computed as one graph but the Python cache stores
        // its K/V in frame units. Preserve that boundary while capturing.
        int64_t frames = 1;
        bool global = false;
    };
    struct cache_input {
        ggml_tensor *tensor;
        const std::vector<float> *src;
        bool f16;
    };
    ggml_context *ctx = nullptr;
    const gguf_loader *file = nullptr;
    const std::unordered_map<const ggml_tensor *, std::vector<float>>
            *f32_weights = nullptr;
    std::unordered_map<std::string, kv_state> *cache = nullptr;
    std::vector<cache_input> cache_inputs;
    std::vector<kv_capture> cache_captures;
    std::string error;
    std::vector<std::pair<ggml_tensor *, int32_t>> position_tensors;
    std::vector<std::pair<ggml_tensor *, float>> scalar_tensors;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> constant_tensors;
    std::unordered_map<ggml_tensor *, ggml_tensor *> f32_casts;
    std::vector<std::pair<std::string, ggml_tensor *>> debug_tensors;
    bool dump_stages = false;
    bool dump_camera_stages = false;
    bool dump_internal = false;
    int64_t rope_special = 6;
    int64_t rope_patch_w = 1;
    int64_t rope_patch_h = 1;
    // Aggregator special-token tables have a leading temporal mode axis:
    // [dim, token_count, first-frame|stream-frame, 1].  This is derived from
    // the stateful global KV cache before constructing each one-frame graph.
    bool stream_token_row = false;
    bool use_dpt_pos = true;
    int cache_scale_frames = 1;
    int cache_window_frames = 4;
    int cache_capture_frames = 1;
    int64_t token_batches = 1;
    int64_t token_frames = 1;
    // LINGBOT_KV_CACHE_F16: store/upload the persistent KV cache in F16,
    // halving the cache footprint that dominates per-frame streaming graphs.
    // Attention stays on the hand-written F32 path (the F16 payload is cast
    // back per layer); "flash" opts into ggml_flash_attn_ext instead. Cache
    // rounding changes either way, so parity gates must be re-run when
    // enabled.
    bool kv_cache_f16 = false;
    bool kv_cache_flash = false;
    // Device-resident KV cache (global layers, f16 non-flash streaming): the
    // F16 payload lives in fixed-capacity tensors owned by model::impl.
    // Reads cast the segment views to F32 in-graph (bit-identical to the
    // legacy path, which uploads the same F16 values and casts them back);
    // each frame persists its K/V with in-graph cpy nodes, so a streaming
    // frame moves no cache bytes over PCIe at all.
    //
    // Payload per layer: pk/pv = {hd, heads, cap_total, 1} F16 laid out as
    // [scale (cap_sc) | live (cap_live frames)] segments, plus sk/sv = F32
    // special segments of cap_sp tokens. The live segment is kept as a
    // contiguous [0, used) frame prefix: eviction copies the oldest frame's
    // leading specials out and compacts through a gallocr temporary, so no
    // in-graph copy ever reads and writes overlapping memory.
    struct res_payload {
        ggml_tensor *pk = nullptr;
        ggml_tensor *pv = nullptr;
        ggml_tensor *sk = nullptr;  // F32 special segment (K)
        ggml_tensor *sv = nullptr;  // F32 special segment (V)
        ggml_context *ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        int64_t cap_sp = 0;    // token capacity of the special segment
        int64_t cap_sc = 0;    // token capacity of the scale segment
        int64_t cap_live = 0;  // FRAME capacity of the live segment
        int64_t sp_count = 0;  // special tokens stored
        int64_t sc_count = 0;  // scale tokens stored
        int64_t used = 0;      // valid frames in the live segment
        int64_t ft = 0;        // frame tokens
        int64_t spf = 0;       // special tokens per frame
    };
    bool kv_resident = false;
    int64_t res_frames_hint = 0;  // whole-run frame count (capacity hint)
    ggml_backend_t res_be = nullptr;
    std::unordered_map<std::string, res_payload> *res_map = nullptr;
    // Pre-phase writes (special-token copies) feed THIS frame's attention, so
    // they expand before every reader; the live shift/append writes touch
    // regions the attention reads, so they expand after every reader.
    std::vector<ggml_tensor *> res_presize;
    std::vector<ggml_tensor *> res_writes;

    void debug_tensor(const std::string &name, ggml_tensor *tensor) {
        bool selected = dump_stages ||
                        (dump_camera_stages && name.rfind("camera.", 0) == 0);
        if (dump_internal) {
            const char *target = g_opt && !g_opt->dump_block.empty()
                                         ? g_opt->dump_block.c_str()
                                         : nullptr;
            if (!target || !*target)
                selected = true;
            else {
                char *end = nullptr;
                const long wanted = std::strtol(target, &end, 10);
                if (std::string(target) == "camera") {
                    selected = name.rfind("camera_head.trunk.", 0) == 0;
                } else {
                    selected = end && *end == '\0' && wanted >= 0 &&
                               name.find("blocks." + std::to_string(wanted)) !=
                                       std::string::npos;
                }
            }
        }
        if (selected && tensor) {
            // Materialize a separate endpoint so allocator reuse cannot alter
            // a value that is still needed by later inference nodes.
            debug_tensors.emplace_back(name, ggml_dup(ctx, f32(tensor)));
        }
    }

    void prepare_cache_for_next_attention(kv_state *state, bool global) {
        // Python appends and evicts before SDPA.  The current frame is not
        // available while this graph is being built, so pre-evict one oldest
        // live frame when appending it would overflow the sliding window.
        if (!state || state->frame_count < cache_scale_frames ||
            state->token_width == 0)
            return;
        const int64_t special_tokens = global ? 6 : 1;
        const int64_t frame_tokens = state->patch_tokens + special_tokens;
        if (frame_tokens <= 0) return;
        const int64_t live_frames = static_cast<int64_t>(state->live_k.size()) /
                                    (frame_tokens * state->token_width);
        if (live_frames < cache_window_frames) return;

        const size_t special_width =
                static_cast<size_t>(special_tokens * state->token_width);
        state->special_k.insert(
                state->special_k.end(), state->live_k.begin(),
                state->live_k.begin() +
                        static_cast<std::ptrdiff_t>(special_width));
        state->special_v.insert(
                state->special_v.end(), state->live_v.begin(),
                state->live_v.begin() +
                        static_cast<std::ptrdiff_t>(special_width));
        const size_t frame_width =
                static_cast<size_t>(frame_tokens * state->token_width);
        state->live_k.erase(state->live_k.begin(),
                            state->live_k.begin() +
                                    static_cast<std::ptrdiff_t>(frame_width));
        state->live_v.erase(state->live_v.begin(),
                            state->live_v.begin() +
                                    static_cast<std::ptrdiff_t>(frame_width));

        state->k.clear();
        state->v.clear();
        state->k.reserve(state->special_k.size() + state->scale_k.size() +
                         state->live_k.size());
        state->v.reserve(state->special_v.size() + state->scale_v.size() +
                         state->live_v.size());
        state->k.insert(state->k.end(), state->special_k.begin(),
                        state->special_k.end());
        state->k.insert(state->k.end(), state->scale_k.begin(),
                        state->scale_k.end());
        state->k.insert(state->k.end(), state->live_k.begin(),
                        state->live_k.end());
        state->v.insert(state->v.end(), state->special_v.begin(),
                        state->special_v.end());
        state->v.insert(state->v.end(), state->scale_v.begin(),
                        state->scale_v.end());
        state->v.insert(state->v.end(), state->live_v.begin(),
                        state->live_v.end());
        state->tokens = static_cast<int64_t>(
                state->k.size() / static_cast<size_t>(state->token_width));
    }

    // ---- device-resident KV cache helpers ----
    ggml_tensor *res_view(const res_payload &rp,
                          ggml_tensor *p,
                          int64_t tok0,
                          int64_t n) const {
        return ggml_view_3d(ctx, p, p->ne[0], p->ne[1], n, p->nb[1], p->nb[2],
                            tok0 * p->nb[2]);
    }
    ggml_tensor *res_live(const res_payload &rp,
                          ggml_tensor *p,
                          int64_t frame0,
                          int64_t frames) const {
        return res_view(rp, p, rp.cap_sc + frame0 * rp.ft, frames * rp.ft);
    }

    // Device-resident read+persist for one global layer on a streaming frame.
    // Overrides k/v with [special | scale | live | fresh] and recomputes
    // old_tokens from the payload counters (the host sidecar goes stale in
    // this mode); the fresh frame is persisted through in-graph cpy nodes.
    void resident_cache(const std::string &key,
                        kv_state &st,
                        ggml_tensor *&k,
                        ggml_tensor *&v,
                        int64_t hd,
                        int64_t heads,
                        int64_t &old_tokens) {
        res_payload &rp = (*res_map)[key];
        const int64_t tw = hd * heads;
        const int64_t ft = k->ne[2];
        const int64_t spf = 6;
        if (!rp.pk) {
            // One-time init: derive the fixed capacities, allocate the
            // payload buffers and seed them from the sidecar (the scale pass
            // has just populated it). Same F32->F16 rounding as the legacy
            // per-frame upload conversion, so the stored bits match.
            const int64_t sc_tok = static_cast<int64_t>(st.scale_k.size()) / tw;
            const int64_t sp_tok =
                    static_cast<int64_t>(st.special_k.size()) / tw;
            const int64_t lv_tok = static_cast<int64_t>(st.live_k.size()) / tw;
            const int64_t total = res_frames_hint > 0 ? res_frames_hint : 512;
            rp.cap_sp = (total + 2) * spf;
            rp.cap_sc = sc_tok;
            rp.cap_live = cache_window_frames + 1;
            const int64_t cap_total = rp.cap_sc + rp.cap_live * ft;
            rp.ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
            rp.pk = ggml_new_tensor_4d(rp.ctx, GGML_TYPE_F16, hd, heads,
                                       cap_total, 1);
            rp.pv = ggml_new_tensor_4d(rp.ctx, GGML_TYPE_F16, hd, heads,
                                       cap_total, 1);
            rp.sk = ggml_new_tensor_4d(rp.ctx, GGML_TYPE_F32, hd, heads,
                                       rp.cap_sp, 1);
            rp.sv = ggml_new_tensor_4d(rp.ctx, GGML_TYPE_F32, hd, heads,
                                       rp.cap_sp, 1);
            ggml_set_input(rp.pk);
            ggml_set_input(rp.pv);
            ggml_set_input(rp.sk);
            ggml_set_input(rp.sv);
            rp.buf = ggml_backend_alloc_ctx_tensors(rp.ctx, res_be);
            if (!rp.buf) {
                error = "resident KV payload allocation failed";
                return;
            }
            rp.ft = ft;
            rp.spf = spf;
            rp.sc_count = sc_tok;
            rp.sp_count = sp_tok;
            rp.used = ft > 0 ? lv_tok / ft : 0;
            auto seed = [&](const std::vector<float> &srcf, ggml_tensor *dst,
                            int64_t tok0, int64_t n_tok, bool to_f16) {
                if (n_tok <= 0) return;
                const int64_t n = n_tok * tw;
                if (to_f16) {
                    std::vector<ggml_fp16_t> h16(static_cast<size_t>(n));
                    for (int64_t i = 0; i < n; ++i)
                        h16[static_cast<size_t>(i)] =
                                ggml_fp32_to_fp16(srcf[static_cast<size_t>(i)]);
                    ggml_backend_tensor_set(dst, h16.data(), tok0 * dst->nb[2],
                                            n * sizeof(ggml_fp16_t));
                } else {
                    ggml_backend_tensor_set(dst, srcf.data(), tok0 * dst->nb[2],
                                            n * sizeof(float));
                }
            };
            seed(st.special_k, rp.sk, 0, sp_tok, false);
            seed(st.special_v, rp.sv, 0, sp_tok, false);
            seed(st.scale_k, rp.pk, 0, sc_tok, true);
            seed(st.scale_v, rp.pv, 0, sc_tok, true);
            seed(st.live_k, rp.pk, rp.cap_sc, lv_tok, true);
            seed(st.live_v, rp.pv, rp.cap_sc, lv_tok, true);
            if (g_opt && g_opt->debug_kv)
                std::fprintf(
                        stderr,
                        "RESIDENT %s cap_sp=%lld cap_sc=%lld cap_live=%lld "
                        "ft=%lld seeded sp=%lld sc=%lld live=%lld\n",
                        key.c_str(), (long long)rp.cap_sp, (long long)rp.cap_sc,
                        (long long)rp.cap_live, (long long)ft,
                        (long long)sp_tok, (long long)sc_tok,
                        (long long)lv_tok);
        }
        // ---- pre-evict (mirrors prepare_cache_for_next_attention): move the
        // oldest frame's leading specials into the F32 segment, then compact
        // the live segment through a gallocr temporary (overlap-free) ----
        if (rp.used >= cache_window_frames) {
            if (rp.sp_count + rp.spf > rp.cap_sp) {
                error = "resident special segment overflow";
                return;
            }
            ggml_tensor *oldest_k = res_live(rp, rp.pk, 0, 1);
            ggml_tensor *oldest_v = res_live(rp, rp.pv, 0, 1);
            auto sp_slice = [&](ggml_tensor *frame) {
                return ggml_view_3d(ctx, frame, frame->ne[0], frame->ne[1],
                                    rp.spf, frame->nb[1], frame->nb[2], 0);
            };
            // Pre-phase: this frame's history read includes the special
            // tokens being copied out, so the copies must run before the
            // attention (they are expanded ahead of every reader).
            res_presize.push_back(ggml_cpy(
                    ctx, ggml_cast(ctx, sp_slice(oldest_k), GGML_TYPE_F32),
                    res_view(rp, rp.sk, rp.sp_count, rp.spf)));
            res_presize.push_back(ggml_cpy(
                    ctx, ggml_cast(ctx, sp_slice(oldest_v), GGML_TYPE_F32),
                    res_view(rp, rp.sv, rp.sp_count, rp.spf)));
            rp.sp_count += rp.spf;
            // The compaction is also pre-phase: this frame's attention reads
            // the live segment at its POST-shift positions [0, used-1), so
            // the shift must complete before the readers. The append below
            // stays in the post phase — its slot lies beyond the read region.
            if (rp.used > 1) {
                const int64_t keep = rp.used - 1;
                res_presize.push_back(ggml_cpy(
                        ctx, ggml_cont(ctx, res_live(rp, rp.pk, 1, keep)),
                        res_live(rp, rp.pk, 0, keep)));
                res_presize.push_back(ggml_cpy(
                        ctx, ggml_cont(ctx, res_live(rp, rp.pv, 1, keep)),
                        res_live(rp, rp.pv, 0, keep)));
            }
            rp.used -= 1;
        }
        old_tokens = rp.sp_count + rp.sc_count + rp.used * ft;
        // ---- read: cast the segment views to F32 (bit-identical to the
        // legacy upload+cast) and concat the fresh frame ----
        ggml_tensor *k_in = k;
        ggml_tensor *v_in = v;
        ggml_tensor *ks = nullptr;
        ggml_tensor *vs = nullptr;
        auto push = [&](ggml_tensor *&acc, ggml_tensor *t) {
            acc = acc ? ggml_concat(ctx, acc, t, 2) : t;
        };
        // Flash mode concatenates everything in F16 (the FA kernel consumes
        // F16 K/V); the default mode casts the F16 payload up to F32 for the
        // hand-written attention. The special segment is stored F32, so its
        // read is cast in whichever direction the mode needs.
        const bool fa = kv_cache_flash;
        ggml_tensor *kf16 = fa ? ggml_cast(ctx, k_in, GGML_TYPE_F16) : nullptr;
        ggml_tensor *vf16 = fa ? ggml_cast(ctx, v_in, GGML_TYPE_F16) : nullptr;
        if (rp.sp_count) {
            push(ks, fa ? ggml_cast(ctx, res_view(rp, rp.sk, 0, rp.sp_count),
                                    GGML_TYPE_F16)
                        : res_view(rp, rp.sk, 0, rp.sp_count));
            push(vs, fa ? ggml_cast(ctx, res_view(rp, rp.sv, 0, rp.sp_count),
                                    GGML_TYPE_F16)
                        : res_view(rp, rp.sv, 0, rp.sp_count));
        }
        if (rp.sc_count) {
            push(ks, fa ? res_view(rp, rp.pk, 0, rp.sc_count)
                        : ggml_cast(ctx, res_view(rp, rp.pk, 0, rp.sc_count),
                                    GGML_TYPE_F32));
            push(vs, fa ? res_view(rp, rp.pv, 0, rp.sc_count)
                        : ggml_cast(ctx, res_view(rp, rp.pv, 0, rp.sc_count),
                                    GGML_TYPE_F32));
        }
        if (rp.used) {
            push(ks, fa ? res_live(rp, rp.pk, 0, rp.used)
                        : ggml_cast(ctx, res_live(rp, rp.pk, 0, rp.used),
                                    GGML_TYPE_F32));
            push(vs, fa ? res_live(rp, rp.pv, 0, rp.used)
                        : ggml_cast(ctx, res_live(rp, rp.pv, 0, rp.used),
                                    GGML_TYPE_F32));
        }
        k = ggml_concat(ctx, ks, fa ? (ggml_tensor *)kf16 : k, 2);
        v = ggml_concat(ctx, vs, fa ? (ggml_tensor *)vf16 : v, 2);
        // ---- persist the fresh frame into the live segment (in-graph; the
        // next frame's segment views read it back) ----
        res_writes.push_back(ggml_cpy(
                ctx,
                fa ? (ggml_tensor *)kf16
                   : (ggml_tensor *)ggml_cast(ctx, k_in, GGML_TYPE_F16),
                res_live(rp, rp.pk, rp.used, 1)));
        res_writes.push_back(ggml_cpy(
                ctx,
                fa ? (ggml_tensor *)vf16
                   : (ggml_tensor *)ggml_cast(ctx, v_in, GGML_TYPE_F16),
                res_live(rp, rp.pv, rp.used, 1)));
        rp.used += 1;
        st.tokens = old_tokens + ft;
    }

    ggml_tensor *scalar(float value) {
        auto *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        scalar_tensors.emplace_back(t, value);
        return t;
    }

    ggml_tensor *find(const std::string &name) const {
        return file->require(name);
    }

    ggml_tensor *find_any(const std::vector<std::string> &names) const {
        for (const auto &name : names) {
            if (auto *t = find(name)) return t;
        }
        return nullptr;
    }

    ggml_tensor *required(const std::vector<std::string> &names,
                          const char *what) {
        if (auto *t = find_any(names)) return t;
        error = std::string("GGUF is missing ") + what + " (tried " +
                names.front() + ")";
        return nullptr;
    }

    ggml_tensor *bias(const std::string &prefix) const {
        return find_any({prefix + ".bias", prefix + ".b"});
    }

    ggml_tensor *f32(ggml_tensor *x) {
        if (!x || x->type == GGML_TYPE_F32) return x;
        if (auto it = f32_casts.find(x); it != f32_casts.end())
            return it->second;
        ggml_tensor *y = nullptr;
        if (f32_weights) {
            auto it = f32_weights->find(x);
            if (it != f32_weights->end()) {
                y = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1],
                                       x->ne[2], x->ne[3]);
                constant_tensors.emplace_back(y, it->second);
            }
        }
        if (!y) {
            // ggml_cast is a graph op and CUDA may legally fuse it back into a
            // quantized matmul.  The parity path needs a real F32 weight buffer
            // so its reduction order matches the PyTorch dequantized reference.
            y = ggml_cont(ctx, ggml_cast(ctx, x, GGML_TYPE_F32));
        }
        f32_casts.emplace(x, y);
        return y;
    }

    ggml_tensor *add_channel_bias(ggml_tensor *x,
                                  ggml_tensor *b,
                                  int64_t channels) {
        if (!b) return x;
        return ggml_add(ctx, x,
                        ggml_reshape_4d(ctx, f32(b), 1, 1, channels, 1));
    }

    ggml_tensor *conv2d(ggml_tensor *x,
                        const std::string &prefix,
                        int stride = 1,
                        int padding = 0,
                        int dilation = 1) {
        auto *w = required({prefix + ".weight", prefix + ".w"}, prefix.c_str());
        if (!w) return nullptr;
        // ggml's generic conv lowers quantized weights to a mul-mat node.
        // Its CUDA default is fp16 accumulation, unlike the fp32 PyTorch
        // reference. Keep convolution reductions in fp32 by making the
        // lowering's left operand fp32; this is required for DPT parity.
        w = f32(w);
        auto *y = ggml_conv_2d(ctx, w, x, stride, stride, padding, padding,
                               dilation, dilation);
        return add_channel_bias(y, bias(prefix), w->ne[3]);
    }

    ggml_tensor *conv_transpose2d(ggml_tensor *x,
                                  const std::string &prefix,
                                  int stride) {
        auto *w = required({prefix + ".weight", prefix + ".w"}, prefix.c_str());
        if (!w) return nullptr;
        w = f32(w);
        auto *y = ggml_conv_transpose_2d_p0(ctx, w, x, stride);
        return add_channel_bias(y, bias(prefix), w->ne[2]);
    }

    ggml_tensor *linear(ggml_tensor *x, const std::string &prefix) {
        auto *w = required({prefix + ".weight", prefix + ".w"}, prefix.c_str());
        if (!w) return nullptr;
        // Camera iterative refinement amplifies q8 dot-product rounding through
        // its AdaLN gate.  Match the deployment reference's dequantize-then-F32
        // GEMM for this bounded head, while retaining q8 kernels for DINO/DPT.
        if (g_opt && g_opt->force_f32_weights ||
            prefix.rfind("camera_head.", 0) == 0)
            w = f32(w);
        auto *y = ggml_mul_mat(ctx, w, x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        // AICore adaptation: mark the output so the Vulkan backend routes it
        // to the exact scalar matmul pipelines (lingbot_merged ggml patch);
        // the name is owner-scoped and invisible to every other task.
        ggml_set_name(y, "lingbot_scalar_proj");
        if (auto *b = bias(prefix)) y = ggml_add(ctx, y, f32(b));
        return y;
    }

    ggml_tensor *norm(ggml_tensor *x,
                      const std::string &prefix,
                      float eps = 1e-6f,
                      bool affine = true) {
        auto *y = ggml_norm(ctx, x, eps);
        if (affine) {
            auto *w = required({prefix + ".weight", prefix + ".gamma"},
                               prefix.c_str());
            if (!w) return nullptr;
            y = ggml_mul(ctx, y, f32(w));
            if (auto *b = find_any({prefix + ".bias", prefix + ".beta"}))
                y = ggml_add(ctx, y, f32(b));
        }
        return y;
    }

    ggml_tensor *maybe_layerscale(ggml_tensor *x, const std::string &prefix) {
        if (auto *g = find_any({prefix + ".gamma", prefix + ".weight"}))
            return ggml_mul(ctx, x, f32(g));
        return x;
    }

    ggml_tensor *rope_constant(int64_t dim,
                               int64_t tokens,
                               int64_t batch,
                               const std::vector<float> &values) {
        auto *t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, 1, tokens, batch);
        constant_tensors.emplace_back(t, values);
        return t;
    }

    ggml_tensor *apply_rope_axis(ggml_tensor *x,
                                 ggml_tensor *cos_t,
                                 ggml_tensor *sin_t,
                                 int64_t offset,
                                 int64_t axis_dim,
                                 int heads,
                                 int64_t tokens,
                                 int64_t batch) {
        const int64_t half = axis_dim / 2;
        auto *part = ggml_view_4d(ctx, x, axis_dim, heads, tokens, batch,
                                  x->nb[1], x->nb[2], x->nb[3],
                                  offset * ggml_type_size(x->type));
        auto *a = ggml_cont(
                ctx, ggml_view_4d(ctx, part, half, heads, tokens, batch,
                                  part->nb[1], part->nb[2], part->nb[3], 0));
        auto *b = ggml_cont(ctx,
                            ggml_view_4d(ctx, part, half, heads, tokens, batch,
                                         part->nb[1], part->nb[2], part->nb[3],
                                         half * ggml_type_size(part->type)));
        auto *rotated = ggml_concat(ctx, ggml_neg(ctx, b), a, 0);
        auto *c = ggml_repeat_4d(ctx, cos_t, axis_dim, heads, tokens, batch);
        auto *s = ggml_repeat_4d(ctx, sin_t, axis_dim, heads, tokens, batch);
        return ggml_add(ctx, ggml_mul(ctx, part, c), ggml_mul(ctx, rotated, s));
    }

    ggml_tensor *rope_2d(ggml_tensor *x,
                         int heads,
                         int64_t tokens,
                         int64_t batch) {
        const int64_t hd = x->ne[0];
        const int64_t axis_dim = hd / 2;
        if (hd % 4 != 0 || axis_dim <= 0) {
            error = "2D RoPE requires head dimension divisible by four";
            return nullptr;
        }
        const int64_t freq_dim = axis_dim / 2;
        std::vector<float> cv(static_cast<size_t>(axis_dim * tokens * batch));
        std::vector<float> sv(cv.size()), ch(cv.size()), sh(cv.size());
        const int64_t per_frame = rope_special + rope_patch_w * rope_patch_h;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t t = 0; t < tokens; ++t) {
                const int64_t local = per_frame > 0 ? t % per_frame : t;
                float py = 0.0f, px = 0.0f;
                if (local >= rope_special) {
                    const int64_t p = local - rope_special;
                    py = static_cast<float>(p / rope_patch_w + 1);
                    px = static_cast<float>(p % rope_patch_w + 1);
                }
                for (int64_t j = 0; j < axis_dim; ++j) {
                    const int64_t f = j < freq_dim ? j : j - freq_dim;
                    const float inv =
                            1.0f /
                            std::pow(100.0f, static_cast<float>(f) / freq_dim);
                    const size_t idx = static_cast<size_t>(
                            j + axis_dim * (t + tokens * b));
                    cv[idx] = std::cos(py * inv);
                    sv[idx] = std::sin(py * inv);
                    ch[idx] = std::cos(px * inv);
                    sh[idx] = std::sin(px * inv);
                }
            }
        }
        auto *cvt = rope_constant(axis_dim, tokens, batch, cv);
        auto *svt = rope_constant(axis_dim, tokens, batch, sv);
        auto *cht = rope_constant(axis_dim, tokens, batch, ch);
        auto *sht = rope_constant(axis_dim, tokens, batch, sh);
        auto *y =
                apply_rope_axis(x, cvt, svt, 0, axis_dim, heads, tokens, batch);
        auto *z = apply_rope_axis(x, cht, sht, axis_dim, axis_dim, heads,
                                  tokens, batch);
        return ggml_concat(ctx, y, z, 0);
    }

    ggml_tensor *add_pos_embed(ggml_tensor *x,
                               int64_t image_width,
                               int64_t image_height,
                               float ratio = 0.1f) {
        const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2],
                      batch = x->ne[3];
        if (c % 4 != 0) return x;
        const float aspect = static_cast<float>(image_width) / image_height;
        const float diag = std::sqrt(aspect * aspect + 1.0f);
        const float span_x = aspect / diag, span_y = 1.0f / diag;
        const float left = -span_x * (w - 1) / w, right = span_x * (w - 1) / w;
        const float top = -span_y * (h - 1) / h, bottom = span_y * (h - 1) / h;
        const int64_t axis = c / 2, bands = axis / 2;
        std::vector<float> data(static_cast<size_t>(w * h * c * batch));
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t yy = 0; yy < h; ++yy) {
                const float py =
                        h == 1 ? 0.0f : top + (bottom - top) * yy / (h - 1);
                for (int64_t xx = 0; xx < w; ++xx) {
                    const float px =
                            w == 1 ? 0.0f
                                   : left + (right - left) * xx / (w - 1);
                    const size_t base = static_cast<size_t>(
                            xx + w * (yy + h * (0 + c * b)));
                    for (int64_t j = 0; j < bands; ++j) {
                        const float inv =
                                1.0f /
                                std::pow(100.0f, static_cast<float>(j) / bands);
                        data[base + j * w * h] = ratio * std::sin(px * inv);
                        data[base + (j + bands) * w * h] =
                                ratio * std::cos(px * inv);
                        data[base + (j + axis) * w * h] =
                                ratio * std::sin(py * inv);
                        data[base + (j + axis + bands) * w * h] =
                                ratio * std::cos(py * inv);
                    }
                }
            }
        auto *pos = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, c, batch);
        constant_tensors.emplace_back(pos, std::move(data));
        return ggml_add(ctx, x, pos);
    }

    ggml_tensor *attention(ggml_tensor *x,
                           const std::string &prefix,
                           int heads,
                           bool use_rope,
                           bool persistent = false,
                           const std::string &cache_key = {}) {
        auto *qkv = linear(x, prefix + ".attn.qkv");
        if (!qkv) return nullptr;
        if (dump_internal) debug_tensor(prefix + ".attn.qkv", qkv);
        const int64_t dim = x->ne[0];
        const int64_t tokens = x->ne[1];
        const int64_t batch = x->ne[2];
        if (dim % heads != 0 || qkv->ne[0] != 3 * dim) {
            error = "invalid attention dimensions in " + prefix;
            return nullptr;
        }
        const int64_t hd = dim / heads;
        auto *q = ggml_view_3d(ctx, qkv, dim, tokens, batch, qkv->nb[1],
                               qkv->nb[2], 0);
        auto *k = ggml_view_3d(ctx, qkv, dim, tokens, batch, qkv->nb[1],
                               qkv->nb[2], dim * ggml_type_size(qkv->type));
        auto *v = ggml_view_3d(ctx, qkv, dim, tokens, batch, qkv->nb[1],
                               qkv->nb[2], 2 * dim * ggml_type_size(qkv->type));
        q = ggml_cont(ctx, q);
        k = ggml_cont(ctx, k);
        v = ggml_cont(ctx, v);
        q = ggml_reshape_4d(ctx, q, hd, heads, tokens, batch);
        k = ggml_reshape_4d(ctx, k, hd, heads, tokens, batch);
        v = ggml_reshape_4d(ctx, v, hd, heads, tokens, batch);
        const bool qk_norm =
                find_any({prefix + ".attn.q_norm.weight"}) != nullptr;
        if (qk_norm) {
            const float qk_eps = prefix.rfind("aggregator.patch_embed.", 0) == 0
                                         ? 1e-6f
                                         : 1e-5f;
            q = ggml_norm(ctx, q, qk_eps);
            k = ggml_norm(ctx, k, qk_eps);
            if (auto *qw = find_any({prefix + ".attn.q_norm.weight"})) {
                q = ggml_mul(ctx, q, f32(qw));
                if (auto *qb = find_any({prefix + ".attn.q_norm.bias"}))
                    q = ggml_add(ctx, q, f32(qb));
            }
            if (auto *kw = find_any({prefix + ".attn.k_norm.weight"})) {
                k = ggml_mul(ctx, k, f32(kw));
                if (auto *kb = find_any({prefix + ".attn.k_norm.bias"}))
                    k = ggml_add(ctx, k, f32(kb));
            }
        }
        if (use_rope) {
            q = rope_2d(f32(q), heads, tokens, batch);
            k = rope_2d(f32(k), heads, tokens, batch);
        }
        if (dump_internal) {
            debug_tensor(prefix + ".attn.q", q);
            debug_tensor(prefix + ".attn.k", k);
            debug_tensor(prefix + ".attn.v", v);
        }
        int64_t old_tokens = 0;
        const std::string &key = cache_key.empty() ? prefix : cache_key;
        if (persistent && cache && batch == 1) {
            auto it = cache->find(key);
            const bool global =
                    prefix.rfind("aggregator.global_blocks.", 0) == 0;
            if (it != cache->end()) {
                // SDPAAttention only applies frame-window eviction when a
                // cache entry has more than one token per frame.  Camera head
                // entries contain one token, so they retain their complete
                // history; global entries use the bounded patch-token window.
                if (global) prepare_cache_for_next_attention(&it->second, true);
                old_tokens = it->second.tokens;
            }
            // The camera trunk keeps an F32 cache even in the F16/flash modes:
            // its 1-token-per-frame history is tiny, and the mirror's
            // CausalAttention has no cache rounding at all — the camera head's
            // iterative AdaLN refinement amplifies any F16 exposure on this
            // path into the pose output (measured 1.5e-06 at the first
            // streaming frame growing to 5.5e-03 over the stream).
            const bool is_camera_trunk =
                    prefix.rfind("camera_head.trunk.", 0) == 0;
            if (kv_cache_f16 && !is_camera_trunk) {
                // F16 cache mode: the persistent cache is uploaded as F16,
                // halving the cache that dominates per-frame streaming
                // graphs. Two attention backends are available:
                // - default: cast the F16 cache back to F32 per layer and run
                //   the hand-written F32 attention below. The cast is a
                //   short-lived intermediate the graph allocator reuses
                //   across layers, so the upload saving survives while
                //   attention keeps the release-grade F32 numerics of the
                //   validated cache path.
                // - "flash": run ggml_flash_attn_ext on F16 K/V (F32 q) for
                //   the streaming frames. The one-shot scale pass keeps the
                //   hand-written F32 path: its bidirectional multi-frame
                //   attention measures ~1e-2-level deviations under the FA
                //   kernel's F16 QK pipeline, while the graph itself fits in
                //   memory anyway. The FA QK pipeline still computes at F16
                //   precision (~1.3e-4 per layer on this model), so "flash"
                //   trades strict-allclose parity for the ~9 GiB graph
                //   reduction that makes 518x378 window=64 fit comfortably.
                const bool res_is_global_layer =
                        prefix.rfind("aggregator.global_blocks.", 0) == 0;
                // Device-resident path: streaming frames only (token_frames
                // == 1), global layers only, non-flash f16 only. The camera
                // trunk keeps its tiny full-history cache on the legacy path.
                const bool res_active = res_is_global_layer && kv_resident &&
                                        token_frames == 1 && old_tokens > 0;
                if (res_active) {
                    resident_cache(key, it->second, k, v, hd, heads,
                                   old_tokens);
                } else if (old_tokens > 0) {
                    auto *pk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hd, heads,
                                                  old_tokens, 1);
                    auto *pv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hd, heads,
                                                  old_tokens, 1);
                    cache_inputs.push_back({pk, &it->second.k, true});
                    cache_inputs.push_back({pv, &it->second.v, true});
                    if (kv_cache_flash) {
                        k = ggml_cast(ctx, k, GGML_TYPE_F16);
                        v = ggml_cast(ctx, v, GGML_TYPE_F16);
                        k = ggml_concat(ctx, pk, k, 2);
                        v = ggml_concat(ctx, pv, v, 2);
                    } else {
                        k = ggml_concat(ctx, ggml_cast(ctx, pk, GGML_TYPE_F32),
                                        k, 2);
                        v = ggml_concat(ctx, ggml_cast(ctx, pv, GGML_TYPE_F32),
                                        v, 2);
                    }
                }
                // Resident layers persist in-graph; no capture readback.
                if (persistent && !res_active) {
                    auto *kc = ggml_cont(
                            ctx,
                            ggml_view_3d(ctx, k, hd, heads, tokens, k->nb[1],
                                         k->nb[2], old_tokens * k->nb[2]));
                    auto *vc = ggml_cont(
                            ctx,
                            ggml_view_3d(ctx, v, hd, heads, tokens, v->nb[1],
                                         v->nb[2], old_tokens * v->nb[2]));
                    cache_captures.push_back(
                            {key, kc, vc, tokens, hd, heads,
                             cache_capture_frames,
                             prefix.rfind("aggregator.global_blocks.", 0) ==
                                     0});
                }
                if (use_rope && !position_tensors.empty())
                    position_tensors.back().second =
                            static_cast<int32_t>(old_tokens);
                if (kv_cache_flash && old_tokens > 0) {
                    // Streaming frames only (see above): run the FA kernel on
                    // the F16 cache + F16 current K/V. ggml_flash_attn_ext
                    // consumes [hd, tokens, heads] layouts: the rope'd q/k/v
                    // live as [hd, heads, tokens], so permute into strided
                    // views (the FA kernels read them with strides). The camera
                    // trunk never reaches this branch (it keeps the F32 cache
                    // path above).
                    auto *qf = ggml_permute(ctx, q, 0, 2, 1, 3);
                    auto *kf = ggml_permute(ctx, k, 0, 2, 1, 3);
                    auto *vf = ggml_permute(ctx, v, 0, 2, 1, 3);
                    auto *fa = ggml_flash_attn_ext(
                            ctx, qf, kf, vf, nullptr,
                            1.0f / std::sqrt(static_cast<float>(hd)), 0.0f,
                            0.0f);
                    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                    // The FA result is materialized as [hd, heads, tokens,
                    // batch] — exactly the token-major layout the hand-written
                    // path produces with its final permute+cont — so the
                    // projection input needs no transpose of its own.
                    auto *out = ggml_reshape_3d(ctx, fa, dim, tokens, batch);
                    if (dump_internal)
                        debug_tensor(prefix + ".attn.value", out);
                    out = linear(out, prefix + ".attn.proj");
                    if (dump_internal) debug_tensor(prefix + ".attn.proj", out);
                    return out;
                }
                // Scale pass (old_tokens == 0) or default mode: fall through
                // to the hand-written F32 attention below with F32 k/v.
            } else {
                if (old_tokens > 0) {
                    auto *pk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads,
                                                  old_tokens, 1);
                    auto *pv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, heads,
                                                  old_tokens, 1);
                    cache_inputs.push_back({pk, &it->second.k, false});
                    cache_inputs.push_back({pv, &it->second.v, false});
                    k = ggml_concat(ctx, pk, k, 2);
                    v = ggml_concat(ctx, pv, v, 2);
                }
                if (persistent) {
                    auto *kc = ggml_cont(
                            ctx,
                            ggml_view_3d(ctx, k, hd, heads, tokens, k->nb[1],
                                         k->nb[2], old_tokens * k->nb[2]));
                    auto *vc = ggml_cont(
                            ctx,
                            ggml_view_3d(ctx, v, hd, heads, tokens, v->nb[1],
                                         v->nb[2], old_tokens * v->nb[2]));
                    cache_captures.push_back(
                            {key, kc, vc, tokens, hd, heads,
                             cache_capture_frames,
                             prefix.rfind("aggregator.global_blocks.", 0) ==
                                     0});
                }
                if (use_rope && !position_tensors.empty())
                    position_tensors.back().second =
                            static_cast<int32_t>(old_tokens);
            }
        }
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        // Exact attention in query chunks: every chunk still softmaxes over
        // the full key axis, so the math is identical to the unchunked form,
        // but the [S_k, qc, heads, batch] score tensor stays in the
        // sub-gigabyte range instead of scaling with S_k x S_q (~4.7 GiB per
        // layer at the 518x378 working point, which is what blocked the
        // parity-grade default mode on 24 GiB cards).
        int64_t q_chunk = (512ull << 20) / (k->ne[1] * q->ne[2] * q->ne[3] *
                                            ggml_type_size(q->type));
        q_chunk = std::max((int64_t)64, std::min(q_chunk, tokens));
        ggml_tensor *out = nullptr;
        for (int64_t q0 = 0; q0 < tokens; q0 += q_chunk) {
            const int64_t qc = std::min(q_chunk, tokens - q0);
            // cont: materialize the chunk so its allocation layout is
            // identical for every q0. The earlier note blaming "Vulkan
            // mul_mat misreads strided-b tensors" was a misdiagnosis: the
            // real culprit was ggml_gallocr stale memory after a cross-size
            // re-reserve on a shared gallocr (backend-independent; see
            // benchmarks/validation_report.md, "Chunked-attention latency"
            // section, and the repro in tools/mul_mat_strided_b_test.cpp). This
            // per-frame graph is built fresh and never re-reserves, so the
            // strided view would also be safe, but the copy is a few MB per
            // chunk and keeps the validated 286-frame path byte-stable.
            auto *q_c =
                    ggml_cont(ctx, ggml_view_4d(ctx, q, hd, qc, q->ne[2],
                                                q->ne[3], q->nb[1], q->nb[2],
                                                q->nb[3], q0 * q->nb[1]));
            auto *scores = ggml_mul_mat(ctx, k, q_c);
            // Attention inputs are presently f32, but persist the precision
            // contract explicitly so a future storage-type optimization cannot
            // silently reintroduce fp16 accumulation here.
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            // AICore adaptation: exact-scalar routing marker (see below).
            ggml_set_name(scores, "lingbot_scalar_scores");
            // Fused scale+softmax: soft_max_ext_inplace computes exp(x*scale -
            // max(x*scale)) in a single pass over the score tensor. Numerically
            // identical to the previous explicit ggml_scale +
            // ggml_soft_max_inplace pair (bit-exact on CPU and CUDA,
            // tools/softmax_scale_test.cpp) and saves one full read+write of
            // the score tensor per chunk (layer latency -7..-22% on CUDA,
            // -12..-16% on Vulkan, tools/chunk_attn_bench.cpp). GCT's streaming
            // cache is causal at frame granularity: a current frame attends to
            // cached frames and to every token in its own frame.
            // ggml_diag_mask_inf would incorrectly make the current frame
            // token-causal, diverging from PyTorch SDPA from the second frame.
            scores = ggml_soft_max_ext_inplace(
                    ctx, scores, nullptr,
                    1.0f / std::sqrt(static_cast<float>(hd)), 0.0f);
            auto *out_c = ggml_mul_mat(ctx, v, scores);
            ggml_mul_mat_set_prec(out_c, GGML_PREC_F32);
            // AICore adaptation: exact-scalar routing marker (lingbot_merged
            // ggml patch) — the P*V product must not take the coopmat2 F32
            // path, which converts operands to F16.
            ggml_set_name(out_c, "lingbot_scalar_attn_out");
            out = out ? ggml_concat(ctx, out, out_c, 1) : out_c;
        }
        if (dump_internal) debug_tensor(prefix + ".attn.value", out);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
        out = ggml_reshape_3d(ctx, out, dim, tokens, batch);
        out = linear(out, prefix + ".attn.proj");
        if (dump_internal) debug_tensor(prefix + ".attn.proj", out);
        return out;
    }

    ggml_tensor *block(ggml_tensor *x,
                       const std::string &prefix,
                       int heads,
                       bool rope,
                       const std::string &cache_key = {}) {
        const float block_eps =
                prefix.rfind("aggregator.patch_embed.", 0) == 0 ? 1e-6f : 1e-5f;
        auto *a = norm(x, prefix + ".norm1", block_eps);
        if (!a) return nullptr;
        if (dump_internal) debug_tensor(prefix + ".norm1", a);
        const bool cache_disabled = g_opt && g_opt->disable_kv_cache;
        const bool persistent =
                !cache_disabled &&
                (prefix.rfind("aggregator.global_blocks.", 0) == 0 ||
                 prefix.rfind("camera_head.trunk.", 0) == 0);
        a = attention(a, prefix, heads, rope, persistent, cache_key);
        if (!a) return nullptr;
        a = maybe_layerscale(a, prefix + ".ls1");
        if (dump_internal) debug_tensor(prefix + ".ls1", a);
        x = ggml_add(ctx, x, a);
        if (dump_internal) debug_tensor(prefix + ".attn_residual", x);
        auto *m = norm(x, prefix + ".norm2", block_eps);
        if (!m) return nullptr;
        if (dump_internal) debug_tensor(prefix + ".norm2", m);
        m = linear(m, prefix + ".mlp.fc1");
        if (!m) return nullptr;
        if (dump_internal) debug_tensor(prefix + ".mlp.fc1", m);
        m = ggml_gelu_erf(ctx, m);
        if (dump_internal) debug_tensor(prefix + ".mlp.gelu", m);
        m = linear(m, prefix + ".mlp.fc2");
        if (!m) return nullptr;
        if (dump_internal) debug_tensor(prefix + ".mlp.fc2", m);
        m = maybe_layerscale(m, prefix + ".ls2");
        if (dump_internal) debug_tensor(prefix + ".ls2", m);
        auto *result = ggml_add(ctx, x, m);
        if (dump_internal) debug_tensor(prefix + ".output", result);
        return result;
    }

    ggml_tensor *repeated_token(const std::vector<std::string> &names,
                                int64_t dim,
                                int64_t count,
                                int64_t batch,
                                bool select_stream_row = false,
                                bool scale_token = false) {
        auto *t = find_any(names);
        if (!t) {
            error = "GGUF is missing special token";
            return nullptr;
        }
        auto *src = f32(t);
        // GGUF stores PyTorch [1, temporal_mode, token_count, dim] tables as
        // [dim, token_count, temporal_mode, 1]. In the initial scale pass,
        // camera/register tokens use row 0 only for the first frame and row 1
        // afterwards. The scale token remains row 0 for every scale frame.
        if (src->ne[0] == dim && src->ne[1] >= count) {
            if (select_stream_row && src->ne[2] >= 2 &&
                token_batches * token_frames == batch) {
                ggml_tensor *expanded = nullptr;
                for (int64_t b = 0; b < token_batches; ++b) {
                    for (int64_t f = 0; f < token_frames; ++f) {
                        const int64_t temporal_row =
                                stream_token_row || (!scale_token && f > 0) ? 1
                                                                            : 0;
                        auto *row = ggml_view_3d(ctx, src, dim, count, 1,
                                                 src->nb[1], src->nb[2],
                                                 temporal_row * src->nb[2]);
                        expanded = expanded ? ggml_concat(ctx, expanded, row, 2)
                                            : row;
                    }
                }
                return expanded;
            }
            const int64_t temporal_row =
                    select_stream_row && stream_token_row && src->ne[2] >= 2
                            ? 1
                            : 0;
            auto *rows = ggml_view_3d(ctx, src, dim, count, 1, src->nb[1],
                                      src->nb[2], temporal_row * src->nb[2]);
            return ggml_reshape_3d(
                    ctx, ggml_repeat_4d(ctx, rows, dim, count, 1, batch), dim,
                    count, batch);
        }
        auto *one = ggml_view_1d(ctx, src, dim, 0);
        return ggml_reshape_3d(ctx,
                               ggml_repeat_4d(ctx, one, dim, count, 1, batch),
                               dim, count, batch);
    }

    ggml_tensor *spatial(ggml_tensor *x,
                         int64_t width,
                         int64_t height,
                         int64_t channels,
                         int64_t batch) {
        auto *r = ggml_reshape_4d(ctx, x, channels, width, height, batch);
        // GGML stores convolution tensors as [W, H, C, N].  The flattened
        // token buffer is [C, W, H, B], so map old axes 2,0,1,3.
        return ggml_cont(ctx, ggml_permute(ctx, r, 2, 0, 1, 3));
    }

    ggml_tensor *residual_conv(ggml_tensor *x, const std::string &prefix) {
        x = ggml_relu(ctx, x);
        // PyTorch's residual unit uses ReLU(inplace=True), so the skip path
        // observes the rectified tensor rather than the pre-activation input.
        auto *orig = x;
        x = conv2d(x, prefix + ".conv1", 1, 1);
        if (!x) return nullptr;
        x = ggml_relu(ctx, x);
        x = conv2d(x, prefix + ".conv2", 1, 1);
        if (!x) return nullptr;
        x = ggml_add(ctx, x, orig);
        return x;
    }

    ggml_tensor *dpt(const std::vector<ggml_tensor *> &features,
                     int64_t width,
                     int64_t height,
                     int64_t channels,
                     int64_t batch,
                     int64_t image_width,
                     int64_t image_height) {
        if (features.size() != 4) {
            error = "DPT requires four intermediate feature maps";
            return nullptr;
        }
        if (!required({"depth_head.norm.weight", "depth_head.norm.gamma"},
                      "DPT norm"))
            return nullptr;
        std::vector<ggml_tensor *> out;
        out.reserve(4);
        const int64_t patch_h = height;
        const int64_t patch_w = width;
        for (int i = 0; i < 4; ++i) {
            auto *x = norm(features[i], "depth_head.norm");
            if (!x) return nullptr;
            x = spatial(x, patch_w, patch_h, channels, batch);
            const std::string p = "depth_head.projects." + std::to_string(i);
            x = conv2d(x, p, 1, 0);
            if (!x) return nullptr;
            debug_tensor("dpt.project" + std::to_string(i), x);
            if (use_dpt_pos) x = add_pos_embed(x, image_width, image_height);
            if (i == 0)
                x = conv_transpose2d(x, "depth_head.resize_layers.0", 4);
            else if (i == 1)
                x = conv_transpose2d(x, "depth_head.resize_layers.1", 2);
            else if (i == 3)
                x = conv2d(x, "depth_head.resize_layers.3", 2, 1);
            if (!x) return nullptr;
            debug_tensor("dpt.resize" + std::to_string(i), x);
            out.push_back(x);
        }
        std::vector<ggml_tensor *> rn(4);
        for (int i = 0; i < 4; ++i) {
            rn[i] = conv2d(
                    out[i],
                    "depth_head.scratch.layer" + std::to_string(i + 1) + "_rn",
                    1, 1);
            if (!rn[i]) return nullptr;
            debug_tensor("dpt.rn" + std::to_string(i), rn[i]);
        }
        auto fuse = [&](int idx, ggml_tensor *coarse, ggml_tensor *skip,
                        int64_t target_w, int64_t target_h) -> ggml_tensor * {
            const std::string p =
                    "depth_head.scratch.refinenet" + std::to_string(idx + 1);
            auto *y = coarse;
            if (skip) {
                auto *r1 = residual_conv(skip, p + ".resConfUnit1");
                if (!r1) return nullptr;
                y = ggml_add(ctx, y, r1);
            }
            auto *r2 = residual_conv(y, p + ".resConfUnit2");
            if (!r2) return nullptr;
            y = ggml_interpolate(
                    ctx, r2, target_w, target_h, r2->ne[2], r2->ne[3],
                    (ggml_scale_mode)(GGML_SCALE_MODE_BILINEAR |
                                      GGML_SCALE_FLAG_ALIGN_CORNERS));
            y = conv2d(y, p + ".out_conv", 1, 0);
            debug_tensor("dpt.refine" + std::to_string(idx), y);
            return y;
        };
        auto *y = fuse(3, rn[3], nullptr, rn[2]->ne[0], rn[2]->ne[1]);
        if (!y) return nullptr;
        y = fuse(2, y, rn[2], rn[1]->ne[0], rn[1]->ne[1]);
        if (!y) return nullptr;
        y = fuse(1, y, rn[1], rn[0]->ne[0], rn[0]->ne[1]);
        if (!y) return nullptr;
        y = fuse(0, y, rn[0], y->ne[0] * 2, y->ne[1] * 2);
        if (!y) return nullptr;
        y = conv2d(y, "depth_head.scratch.output_conv1", 1, 1);
        if (!y) return nullptr;
        debug_tensor("dpt.output_conv1", y);
        y = ggml_interpolate(ctx, y, image_width, image_height, y->ne[2], batch,
                             (ggml_scale_mode)(GGML_SCALE_MODE_BILINEAR |
                                               GGML_SCALE_FLAG_ALIGN_CORNERS));
        if (use_dpt_pos) y = add_pos_embed(y, image_width, image_height);
        y = conv2d(y, "depth_head.scratch.output_conv2.0", 1, 1);
        if (!y) return nullptr;
        debug_tensor("dpt.output_conv2_0", y);
        y = ggml_relu(ctx, y);
        y = conv2d(y, "depth_head.scratch.output_conv2.2", 1, 0);
        if (!y) return nullptr;
        debug_tensor("dpt.logits", y);
        return y;
    }
};

}  // namespace

struct model::impl {
    backend be;
    std::string backend_name;
    int threads_resolved = 0;
    // Load-time snapshot of the caller's options; g_opt points here.
    model_options opts;
    gguf_loader file;
    ggml_backend_buffer_t weights = nullptr;
    ggml_backend_buffer_t host_weights = nullptr;
    ggml_context *device_ctx = nullptr;
    std::unordered_map<const ggml_tensor *, std::vector<float>> f32_weights;
    // Host f32 copy of aggregator.patch_embed.pos_embed, captured at load time
    // before tensor data pointers are repointed at device buffers. It is the
    // only graph constant that depends on the input resolution, so a
    // reduced-resolution graph resamples it on the host (see model::infer).
    std::vector<float> pos_embed_host;
    model_info info;
    std::string error;
    std::unordered_map<std::string, graph_builder::kv_state> kv_cache;
    uint64_t dump_index = 0;
    uint64_t inference_index = 0;
    bool direct_scale_pass = false;
    // Per-frame streaming hook (see model::set_frame_callback). Active only
    // for the top-level streaming dispatch; the internal scale-pass recursion
    // runs with the cursor suspended so it can emit every scale frame itself.
    std::function<bool(int, const output &)> frame_cb;
    int frame_cursor = 0;

    // Pooled graph allocator. A fresh gallocr per frame costs a full
    // cudaMalloc + cudaFree of the ~9 GB scratch pool on every infer, which
    // measured ~6-7 s of driver-side wall time per streamed frame (GPU
    // utilization 15%). Once the streaming cache reaches its window bound the
    // graph shape is IDENTICAL frame to frame, and repeated same-size
    // alloc_graph calls on one gallocr are the verified-safe pattern (the
    // stale-memory defect only fires on cross-size re-reserve). The key is a
    // fingerprint of every node's shape/type; on mismatch the pooled gallocr
    // is rebuilt (growth phase only) instead of re-reserved in place.
    ggml_gallocr_t pooled_allocr = nullptr;
    size_t pooled_key = 0;
    // Device-resident KV payloads, keyed by layer cache key (see
    // graph_builder::res_payload). Owns the payload contexts and buffers.
    std::unordered_map<std::string, graph_builder::res_payload> res_cache;
};
namespace {

// FNV-1a fingerprint of the full graph shape (node count, per-node type and
// extents). Same fingerprint => same gallocr pool requirement => same-size
// reuse without a cross-size re-reserve.
size_t graph_shape_fingerprint(ggml_cgraph *g) {
    size_t h = 1469598103934665603ull;
    auto mix = [&](size_t v) {
        h ^= v + 0x9e3779b97f4a7c15ull;
        h *= 1099511628211ull;
    };
    const int n = ggml_graph_n_nodes(g);
    mix((size_t)n);
    for (int i = 0; i < n; ++i) {
        ggml_tensor *t = ggml_graph_node(g, i);
        if (!t) {
            mix(0);
            continue;
        }
        mix((size_t)t->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) mix((size_t)t->ne[d]);
    }
    return h;
}

}  // namespace

namespace {

bool dump_enabled_for_inference(bool enabled,
                                int dump_at,
                                uint64_t inference_index) {
    if (!enabled) return false;
    if (dump_at < 0) return true;
    return static_cast<uint64_t>(dump_at) == inference_index;
}

void append_token_range(std::vector<float> *dst,
                        const std::vector<float> &src,
                        int64_t first_token,
                        int64_t token_count,
                        int64_t token_width) {
    const size_t begin = static_cast<size_t>(first_token * token_width);
    const size_t end = begin + static_cast<size_t>(token_count * token_width);
    if (end > src.size()) {
        std::fprintf(stderr,
                     "LingBot KV capture range exceeds source: begin=%zu "
                     "end=%zu size=%zu\n",
                     begin, end, src.size());
        std::abort();
    }
    dst->insert(dst->end(), src.begin() + static_cast<std::ptrdiff_t>(begin),
                src.begin() + static_cast<std::ptrdiff_t>(end));
}

void append_cache_capture(graph_builder::kv_state *state,
                          const graph_builder::kv_capture &capture,
                          const std::vector<float> &frame_k,
                          const std::vector<float> &frame_v,
                          int scale_frames,
                          int window_frames) {
    // Phase 1 executes all scale frames in one graph for bidirectional global
    // attention. The persisted Python cache is nevertheless frame-indexed:
    // split this capture before applying the normal scale/live eviction rules.
    if (capture.frames > 1) {
        if (capture.tokens % capture.frames != 0) {
            std::fprintf(stderr,
                         "LingBot KV capture is not divisible into frames: "
                         "tokens=%lld frames=%lld\n",
                         static_cast<long long>(capture.tokens),
                         static_cast<long long>(capture.frames));
            std::abort();
        }
        graph_builder::kv_capture one = capture;
        one.tokens /= one.frames;
        one.frames = 1;
        const size_t frame_width =
                static_cast<size_t>(one.tokens * one.head_dim * one.heads);
        for (int64_t frame = 0; frame < capture.frames; ++frame) {
            const size_t offset = static_cast<size_t>(frame) * frame_width;
            if (offset + frame_width > frame_k.size() ||
                offset + frame_width > frame_v.size()) {
                std::fprintf(stderr,
                             "LingBot KV capture frame range exceeds source\n");
                std::abort();
            }
            std::vector<float> one_k(
                    frame_k.begin() + static_cast<std::ptrdiff_t>(offset),
                    frame_k.begin() +
                            static_cast<std::ptrdiff_t>(offset + frame_width));
            std::vector<float> one_v(
                    frame_v.begin() + static_cast<std::ptrdiff_t>(offset),
                    frame_v.begin() +
                            static_cast<std::ptrdiff_t>(offset + frame_width));
            append_cache_capture(state, one, one_k, one_v, scale_frames,
                                 window_frames);
        }
        return;
    }
    const int64_t token_width = capture.head_dim * capture.heads;
    if (state->token_width != 0 && state->token_width != token_width)
        *state = {};
    state->token_width = token_width;

    // CameraCausalHead has exactly one token per frame.  PyTorch deliberately
    // bypasses its KV eviction branch for that layout, retaining persistent
    // cross-call history for every iteration and trunk layer.
    if (!capture.global) {
        state->k.insert(state->k.end(), frame_k.begin(), frame_k.end());
        state->v.insert(state->v.end(), frame_v.begin(), frame_v.end());
        state->tokens += capture.tokens;
        ++state->frame_count;
        state->patch_tokens = 0;
        return;
    }

    // Match SDPAAttention._apply_kv_cache_eviction exactly.  Regular cache
    // entries are complete frames.  Only special tokens of frames evicted from
    // the live window move into the separate prefix cache.
    const int64_t special_tokens = capture.global && capture.tokens > 6 ? 6 : 1;
    if (state->frame_count < scale_frames) {
        append_token_range(&state->scale_k, frame_k, 0, capture.tokens,
                           token_width);
        append_token_range(&state->scale_v, frame_v, 0, capture.tokens,
                           token_width);
    } else {
        append_token_range(&state->live_k, frame_k, 0, capture.tokens,
                           token_width);
        append_token_range(&state->live_v, frame_v, 0, capture.tokens,
                           token_width);
        const int64_t live_frames = static_cast<int64_t>(state->live_k.size()) /
                                    (capture.tokens * token_width);
        const int64_t evicted_frames = live_frames - window_frames;
        if (evicted_frames > 0) {
            if (special_tokens > 0) {
                for (int64_t i = 0; i < evicted_frames; ++i) {
                    append_token_range(&state->special_k, state->live_k,
                                       i * capture.tokens, special_tokens,
                                       token_width);
                    append_token_range(&state->special_v, state->live_v,
                                       i * capture.tokens, special_tokens,
                                       token_width);
                }
            }
            const size_t erase_count = static_cast<size_t>(
                    evicted_frames * capture.tokens * token_width);
            state->live_k.erase(
                    state->live_k.begin(),
                    state->live_k.begin() +
                            static_cast<std::ptrdiff_t>(erase_count));
            state->live_v.erase(
                    state->live_v.begin(),
                    state->live_v.begin() +
                            static_cast<std::ptrdiff_t>(erase_count));
        }
    }
    ++state->frame_count;
    state->patch_tokens = capture.tokens - special_tokens;

    state->k.clear();
    state->v.clear();
    state->k.reserve(state->special_k.size() + state->scale_k.size() +
                     state->live_k.size());
    state->v.reserve(state->special_v.size() + state->scale_v.size() +
                     state->live_v.size());
    state->k.insert(state->k.end(), state->special_k.begin(),
                    state->special_k.end());
    state->k.insert(state->k.end(), state->scale_k.begin(),
                    state->scale_k.end());
    state->k.insert(state->k.end(), state->live_k.begin(), state->live_k.end());
    state->v.insert(state->v.end(), state->special_v.begin(),
                    state->special_v.end());
    state->v.insert(state->v.end(), state->scale_v.begin(),
                    state->scale_v.end());
    state->v.insert(state->v.end(), state->live_v.begin(), state->live_v.end());
    state->tokens = static_cast<int64_t>(state->k.size() /
                                         static_cast<size_t>(token_width));
    if (g_opt && g_opt->debug_kv &&
        capture.key == "aggregator.global_blocks.0") {
        std::fprintf(stderr,
                     "LingBot KV frame=%lld visible_tokens=%lld special=%zu "
                     "scale=%zu live=%zu\n",
                     static_cast<long long>(state->frame_count),
                     static_cast<long long>(state->tokens),
                     state->special_k.size() / static_cast<size_t>(token_width),
                     state->scale_k.size() / static_cast<size_t>(token_width),
                     state->live_k.size() / static_cast<size_t>(token_width));
    }
}

}  // namespace

namespace {

// Pillow's bicubic filter (a = -0.5): the kernel PyTorch evaluates for
// antialias=True resampling, distinct from the -0.75 Keys kernel of the
// non-antialias bicubic path.
float bicubic_aa_filter(float x) {
    constexpr float a = -0.5f;
    if (x < 0.0f) x = -x;
    if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
    return 0.0f;
}

// Normalized windowed weights for one output index, replicating
// upsample_antialias::_compute_weights_span/_compute_weights: the support
// grows with the downsample ratio, the span truncates toward zero and is
// clamped to the input extent, and the tap weights sum to one.
void aa_axis_weights(int dst,
                     int in_size,
                     float scale,
                     int *start,
                     std::vector<float> *weights) {
    const float support = scale >= 1.0f ? 2.0f * scale : 2.0f;
    const float center = scale * (dst + 0.5f);
    *start = std::max(static_cast<int>(center - support + 0.5f), 0);
    const int count =
            std::min(static_cast<int>(center + support + 0.5f), in_size) -
            *start;
    weights->assign(count, 0.0f);
    const float invscale = scale >= 1.0f ? 1.0f / scale : 1.0f;
    float total = 0.0f;
    for (int j = 0; j < count; ++j) {
        const float w =
                bicubic_aa_filter((j + *start - center + 0.5f) * invscale);
        (*weights)[j] = w;
        total += w;
    }
    if (total != 0.0f) {
        for (float &w : *weights) w /= total;
    }
}

// Resample the exported DINO patch table (rows 1..m*m over an m x m patch
// grid, row 0 is CLS) to the patch grid of the running input. This replicates
// DINOv2's interpolate_pos_encoding, i.e. nn.functional.interpolate with
// mode="bicubic", antialias=True, align_corners=False: PyTorch executes
// upsample_gen2d_aa_out_frame with BicubicFilterFunctor, so every
// accumulation below stays in float32 and follows the same horizontal-then-
// vertical order. torch flattens the resampled (H_out=w0, W_out=h0) grid
// row-major, so token t reads grid row t / h0 and column t % h0; the two
// orders coincide for the square inputs every benchmark profile uses.
// src is token-major [token][dim].
std::vector<float> interpolate_dino_pos_table(const std::vector<float> &src,
                                              int64_t dim,
                                              int m,
                                              int out_w,
                                              int out_h) {
    // torch flattens the resampled (out_h, out_w) grid row-major over the
    // width: token t reads grid row t / out_w and column t % out_w. The row
    // axis resamples m -> out_h and the column axis m -> out_w; these only
    // coincide for square patch grids (392/518), so the denominators must
    // not be swapped or non-square inputs like 518x378 interpolate the
    // wrong axes.
    const float scale_y = static_cast<float>(m) / static_cast<float>(out_h);
    const float scale_x = static_cast<float>(m) / static_cast<float>(out_w);
    std::vector<float> dst(static_cast<size_t>(1 + out_w * out_h) * dim, 0.0f);
    std::copy_n(src.begin(), dim, dst.begin());
    std::vector<float> wy, wx, horiz(static_cast<size_t>(dim));
    int ymin = 0, xmin = 0;
    for (int t = 0; t < out_w * out_h; ++t) {
        const int i = t / out_w;
        const int j = t % out_w;
        aa_axis_weights(i, m, scale_y, &ymin, &wy);
        aa_axis_weights(j, m, scale_x, &xmin, &wx);
        float *out_row = dst.data() + static_cast<size_t>(1 + t) * dim;
        for (int y = 0; y < static_cast<int>(wy.size()); ++y) {
            const float *row =
                    src.data() +
                    static_cast<size_t>(1 + (ymin + y) * m + xmin) * dim;
            std::fill(horiz.begin(), horiz.end(), 0.0f);
            for (int x = 0; x < static_cast<int>(wx.size()); ++x) {
                const float w = wx[x];
                const float *tap = row + static_cast<size_t>(x) * dim;
                for (int64_t c = 0; c < dim; ++c) horiz[c] += w * tap[c];
            }
            const float wyv = wy[y];
            for (int64_t c = 0; c < dim; ++c) out_row[c] += wyv * horiz[c];
        }
    }
    return dst;
}

}  // namespace

model::model() : p_(new impl) {}

void model::set_frame_callback(std::function<bool(int, const output &)> cb) {
    p_->frame_cb = std::move(cb);
    p_->frame_cursor = 0;
}
model::~model() = default;
model::model(model &&) noexcept = default;
model &model::operator=(model &&) noexcept = default;

bool model::load(const std::string &path,
                 const std::string &backend_name,
                 int threads,
                 const model_options &options) {
    release();
    p_->opts =
            options;  // own the snapshot: the caller's struct may be temporary
    g_opt = &p_->opts;
    if (!p_->file.open(path)) {
        p_->error = p_->file.error();
        return false;
    }
    if (!p_->be.init(backend_name, threads)) {
        p_->error = p_->be.error();
        return false;
    }
    g_opt = &p_->opts;  // backend init may run on a first-use worker path
    p_->backend_name = backend_name;
    p_->threads_resolved = threads;
    p_->info.image_size = p_->file.i32("lingbot-map.image_size", 518);
    p_->info.patch_size = p_->file.i32("lingbot-map.patch_size", 14);
    p_->info.embed_dim = p_->file.i32("lingbot-map.embedding_length", 1024);
    p_->info.block_count = p_->file.i32("lingbot-map.block_count", 24);
    p_->info.output_dim = p_->file.i32("lingbot-map.output_dim", 9);
    p_->info.weight_type = p_->file.str("lingbot-map.weight_type", "unknown");
    p_->info.graph_version =
            p_->file.str("lingbot-map.graph_version", "missing");
    if (p_->info.graph_version != "gct-v1-contract") {
        p_->error = "GGUF graph contract is missing or incompatible";
        return false;
    }
    void *base = ggml_get_mem_buffer(p_->file.context());
    const size_t bytes = ggml_get_mem_size(p_->file.context());
    p_->host_weights = ggml_backend_cpu_buffer_from_ptr(base, bytes);
    if (!p_->host_weights) {
        p_->error = "failed to bind GGUF weights to host memory";
        return false;
    }
    for (ggml_tensor *t = ggml_get_first_tensor(p_->file.context()); t;
         t = ggml_get_next_tensor(p_->file.context(), t)) {
        t->buffer = p_->host_weights;
        if (p_->pos_embed_host.empty() && t->data &&
            std::strcmp(ggml_get_name(t), "aggregator.patch_embed.pos_embed") ==
                    0) {
            const auto *traits = ggml_get_type_traits(t->type);
            std::vector<float> values(static_cast<size_t>(t->ne[0]) * t->ne[1]);
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(values.data(), t->data,
                            values.size() * sizeof(float));
            } else if (!traits->is_quantized && traits->to_float) {
                for (int64_t r = 0; r < t->ne[1]; ++r) {
                    traits->to_float(
                            static_cast<const char *>(t->data) + r * t->nb[1],
                            values.data() + static_cast<size_t>(r) * t->ne[0],
                            t->ne[0]);
                }
            } else {
                values.clear();
            }
            p_->pos_embed_host = std::move(values);
        }
    }
    p_->f32_weights.clear();
    if (g_opt && g_opt->force_f32_weights) {
        for (ggml_tensor *s = ggml_get_first_tensor(p_->file.context()); s;
             s = ggml_get_next_tensor(p_->file.context(), s)) {
            const auto *traits = ggml_get_type_traits(s->type);
            if (!traits->is_quantized || !s->data) continue;
            const int64_t rows = s->ne[1] * s->ne[2] * s->ne[3];
            std::vector<float> values(static_cast<size_t>(s->ne[0] * rows));
            for (int64_t row = 0; row < rows; ++row) {
                const auto *src = static_cast<const char *>(s->data) +
                                  static_cast<size_t>(row) * s->nb[1];
                traits->to_float(src, values.data() + row * s->ne[0], s->ne[0]);
            }
            p_->f32_weights.emplace(s, std::move(values));
        }
    }
    if (backend_name == "cpu") {
        p_->weights = p_->host_weights;
        p_->host_weights = nullptr;
    } else {
        size_t n_tensors = 0;
        for (ggml_tensor *t = ggml_get_first_tensor(p_->file.context()); t;
             t = ggml_get_next_tensor(p_->file.context(), t))
            ++n_tensors;
        ggml_init_params dp = {ggml_tensor_overhead() * (n_tensors + 8),
                               nullptr, true};
        p_->device_ctx = ggml_init(dp);
        if (!p_->device_ctx) {
            p_->error = "failed to create device weight context";
            return false;
        }
        for (ggml_tensor *s = ggml_get_first_tensor(p_->file.context()); s;
             s = ggml_get_next_tensor(p_->file.context(), s)) {
            auto *d = ggml_dup_tensor(p_->device_ctx, s);
            ggml_set_name(d, ggml_get_name(s));
        }
        p_->weights =
                ggml_backend_alloc_ctx_tensors(p_->device_ctx, p_->be.get());
        if (!p_->weights) {
            p_->error = "failed to allocate device weight tensors";
            return false;
        }
        for (ggml_tensor *s = ggml_get_first_tensor(p_->file.context()); s;
             s = ggml_get_next_tensor(p_->file.context(), s)) {
            auto *d = ggml_get_tensor(p_->device_ctx, ggml_get_name(s));
            ggml_backend_tensor_set(d, s->data, 0, ggml_nbytes(s));
            s->buffer = d->buffer;
            s->data = d->data;
            for (int k = 0; k < GGML_MAX_DIMS; ++k) s->nb[k] = d->nb[k];
        }
        ggml_backend_synchronize(p_->be.get());
    }
    return true;
}

void model::release() {
    if (!p_) return;
    if (p_->pooled_allocr) {
        ggml_gallocr_free(p_->pooled_allocr);
        p_->pooled_allocr = nullptr;
        p_->pooled_key = 0;
    }
    if (p_->weights) ggml_backend_buffer_free(p_->weights);
    if (p_->host_weights) ggml_backend_buffer_free(p_->host_weights);
    if (p_->device_ctx) ggml_free(p_->device_ctx);
    p_->weights = nullptr;
    p_->host_weights = nullptr;
    p_->device_ctx = nullptr;
    p_->pos_embed_host.clear();
    p_->be.release();
    p_->backend_name.clear();
    p_->file.close();
    p_->error.clear();
    p_->info = {};
    p_->kv_cache.clear();
    for (auto &item : p_->res_cache) {
        if (item.second.buf) ggml_backend_buffer_free(item.second.buf);
        if (item.second.ctx) ggml_free(item.second.ctx);
    }
    p_->res_cache.clear();
    p_->dump_index = 0;
}

void model::reset_cache() {
    if (p_) p_->kv_cache.clear();
}

bool model::infer(const float *images,
                  int batch,
                  int frames,
                  int channels,
                  int height,
                  int width,
                  output *result) {
    if (!images || !result || !p_->be.get() || !p_->weights) {
        p_->error = "model is not loaded or output is null";
        return false;
    }
    g_opt = &p_->opts;  // rebind this thread's options snapshot before graph
                        // work
    if (batch <= 0 || frames <= 0 || channels != 3 || height <= 0 ||
        width <= 0) {
        p_->error = "expected images with shape [B,S,3,H,W]";
        return false;
    }
    if (height % p_->info.patch_size || width % p_->info.patch_size) {
        p_->error = "image dimensions must be divisible by patch_size";
        return false;
    }

    // Phase 1 of the official streaming contract evaluates its initial scale
    // frames as one bidirectional block. Later frames are single-frame calls.
    // The constrained one-frame profile remains the default.
    if (frames > 1 && batch == 1 && !p_->direct_scale_pass) {
        result->batch = 1;
        result->frames = frames;
        result->height = height;
        result->width = width;
        result->pose_enc.clear();
        result->depth.clear();
        result->depth_conf.clear();
        result->c2w.clear();
        result->intrinsics.clear();
        const size_t frame_stride =
                static_cast<size_t>(channels) * height * width;
        const int opt_scale_frames = g_opt->num_scale_frames > 0
                                             ? g_opt->num_scale_frames
                                             : g_opt->kv_cache_scale;
        const int scale_frames = std::min(frames, opt_scale_frames);
        // Emit one callback per completed frame (see
        // model::set_frame_callback). The first argument is the frame index
        // *inside `src`* (the scale pass hands a multi-frame output, streaming
        // frames hand a single-frame one whose internal index is always 0); the
        // callback receives the global frame cursor. Slicing with the global
        // index instead would read past the single-frame vectors.
        auto emit_frame = [&](int src_idx, const output &src) {
            if (!p_->frame_cb) return true;
            output one;
            one.batch = 1;
            one.frames = 1;
            one.height = height;
            one.width = width;
            one.pose_enc.assign(src.pose_enc.begin() + src_idx * 9,
                                src.pose_enc.begin() + src_idx * 9 + 9);
            const size_t plane = static_cast<size_t>(height) * width;
            one.depth.assign(src.depth.begin() + src_idx * plane,
                             src.depth.begin() + (src_idx + 1) * plane);
            one.depth_conf.assign(
                    src.depth_conf.begin() + src_idx * plane,
                    src.depth_conf.begin() + (src_idx + 1) * plane);
            one.c2w.assign(src.c2w.begin() + src_idx * 16,
                           src.c2w.begin() + src_idx * 16 + 16);
            one.intrinsics.assign(src.intrinsics.begin() + src_idx * 4,
                                  src.intrinsics.begin() + src_idx * 4 + 4);
            if (!p_->frame_cb(p_->frame_cursor++, one)) {
                p_->error = "frame callback aborted";
                return false;
            }
            return true;
        };
        int first_frame = 0;
        if (scale_frames > 1 && p_->kv_cache.empty()) {
            output scale;
            p_->direct_scale_pass = true;
            const bool ok = infer(images, 1, scale_frames, channels, height,
                                  width, &scale);
            p_->direct_scale_pass = false;
            if (!ok) return false;
            result->pose_enc.insert(result->pose_enc.end(),
                                    scale.pose_enc.begin(),
                                    scale.pose_enc.end());
            result->depth.insert(result->depth.end(), scale.depth.begin(),
                                 scale.depth.end());
            result->depth_conf.insert(result->depth_conf.end(),
                                      scale.depth_conf.begin(),
                                      scale.depth_conf.end());
            result->c2w.insert(result->c2w.end(), scale.c2w.begin(),
                               scale.c2w.end());
            result->intrinsics.insert(result->intrinsics.end(),
                                      scale.intrinsics.begin(),
                                      scale.intrinsics.end());
            for (int f = 0; f < scale_frames; ++f)
                if (!emit_frame(f, scale)) return false;
            first_frame = scale_frames;
        }
        // The scale-pass graph reserve is dead weight from here on: streaming
        // graphs have different shapes and would rebuild the pool anyway.
        // Release it before the resident payload allocates, so backends
        // without an allocator cache above the driver (Vulkan) can fit both.
        if (p_->pooled_allocr) {
            ggml_gallocr_free(p_->pooled_allocr);
            p_->pooled_allocr = nullptr;
            p_->pooled_key = 0;
        }
        for (int f = first_frame; f < frames; ++f) {
            output one;
            if (!infer(images + static_cast<size_t>(f) * frame_stride, 1, 1,
                       channels, height, width, &one))
                return false;
            result->pose_enc.insert(result->pose_enc.end(),
                                    one.pose_enc.begin(), one.pose_enc.end());
            result->depth.insert(result->depth.end(), one.depth.begin(),
                                 one.depth.end());
            result->depth_conf.insert(result->depth_conf.end(),
                                      one.depth_conf.begin(),
                                      one.depth_conf.end());
            result->c2w.insert(result->c2w.end(), one.c2w.begin(),
                               one.c2w.end());
            result->intrinsics.insert(result->intrinsics.end(),
                                      one.intrinsics.begin(),
                                      one.intrinsics.end());
            if (!emit_frame(0, one)) return false;
        }
        return true;
    }

    // A full stream can be inspected without serialising every intermediate
    // tensor.  LINGBOT_DUMP_AT selects a zero-based leaf inference index.
    const uint64_t inference_index = p_->inference_index++;
    const bool dump_stages = dump_enabled_for_inference(
            g_opt->dump_stages, g_opt->dump_at, inference_index);
    // Per-frame progress chatter stays off unless debugging is requested;
    // the GUI relays stderr on failure and this line would otherwise bury
    // the real error under one row per frame.
    if (g_opt->debug_infer) {
        std::fprintf(stderr, "DBG infer index=%llu frames=%d dump_stages=%d\n",
                     (unsigned long long)inference_index, frames,
                     (int)dump_stages);
    }
    const bool dump_camera_stages = dump_enabled_for_inference(
            g_opt->dump_camera_stages, g_opt->dump_at, inference_index);
    const bool dump_features = dump_enabled_for_inference(
            !g_opt->dump_features_dir.empty(), g_opt->dump_at, inference_index);

    const int64_t bs = static_cast<int64_t>(batch) * frames;
    const int64_t ph = height / p_->info.patch_size;
    const int64_t pw = width / p_->info.patch_size;
    const int64_t patches = ph * pw;
    const int64_t dim = p_->info.embed_dim;
    const int heads = std::max(1, p_->file.i32("lingbot-map.num_heads", 16));
    const int layers = std::max(1, p_->info.block_count);

    ggml_init_params params = {
            ggml_tensor_overhead() * 200000 +
                    static_cast<size_t>(bs * height * width * channels *
                                        sizeof(float)),
            nullptr, true};
    ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        p_->error = "ggml context allocation failed";
        return false;
    }
    graph_builder g;
    g.ctx = ctx;
    g.file = &p_->file;
    g.f32_weights = p_->f32_weights.empty() ? nullptr : &p_->f32_weights;
    g.cache = &p_->kv_cache;
    g.stream_token_row = !p_->kv_cache.empty();
    g.rope_special = 6;
    g.rope_patch_w = pw;
    g.rope_patch_h = ph;
    g.cache_scale_frames = g_opt->kv_cache_scale;
    g.cache_window_frames = g_opt->kv_cache_window;
    g.kv_cache_f16 = g_opt->kv_f16 != kv_f16_mode::none;
    g.kv_cache_flash = g_opt->kv_f16 == kv_f16_mode::flash;
    // Device-resident KV cache: the streaming global-layer payload lives in
    // fixed-capacity device tensors written in-graph (no per-frame cache
    // conversion, upload or capture readback). Default on for the non-flash
    // F16 mode; kv_resident=false falls back to the host-sidecar path.
    g.kv_resident = g.kv_cache_f16 && g_opt->kv_resident;
    g.res_frames_hint =
            g_opt->kv_total_frames > 0 ? g_opt->kv_total_frames : 512;
    g.res_map = &p_->res_cache;
    g.res_be = p_->be.get();
    g.cache_capture_frames = batch == 1 ? frames : 1;
    g.token_batches = batch;
    g.token_frames = frames;
    g.dump_stages = dump_stages;
    g.dump_camera_stages = dump_camera_stages;
    g.dump_internal = dump_enabled_for_inference(
            g_opt->dump_internal, g_opt->dump_at, inference_index);
    if (!g_opt->use_dpt_pos) g.use_dpt_pos = false;
    auto *input =
            ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height, channels, bs);

    // GCT normalizes RGB with ImageNet statistics before DINO/conv patch
    // embedding.
    auto *mean = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 3, 1);
    auto *stdev = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 3, 1);
    const float mean_v[3] = {0.485f, 0.456f, 0.406f};
    const float std_v[3] = {0.229f, 0.224f, 0.225f};
    auto *mean_full = ggml_repeat_4d(ctx, mean, width, height, channels, bs);
    auto *std_full = ggml_repeat_4d(ctx, stdev, width, height, channels, bs);
    auto *normed = ggml_div(ctx, ggml_sub(ctx, input, mean_full), std_full);

    auto *patch_w =
            g.required({"aggregator.patch_embed.patch_embed.proj.weight",
                        "aggregator.patch_embed.proj.weight"},
                       "patch embedding weight");
    if (!patch_w) {
        p_->error = g.error;
        ggml_free(ctx);
        return false;
    }
    // AICore adaptation: the upstream LINGBOT_FORCE_F32_WEIGHTS environment
    // debug toggle is gone; force_f32_weights remains a plain model_option.
    if (g_opt && g_opt->force_f32_weights) patch_w = g.f32(patch_w);
    auto *patch = ggml_conv_2d_direct(ctx, patch_w, normed, p_->info.patch_size,
                                      p_->info.patch_size, 0, 0, 1, 1);
    if (auto *patch_b =
                g.find_any({"aggregator.patch_embed.patch_embed.proj.bias",
                            "aggregator.patch_embed.proj.bias"})) {
        patch = ggml_add(ctx, patch,
                         ggml_reshape_4d(ctx, g.f32(patch_b), 1, 1, dim, 1));
    }
    // ggml_conv_2d labels its output [W,H,C,B]. Convert it to [C,W,H,B]
    // before flattening spatial positions. Keeping B separate is essential:
    // folding it into C before a transpose reorders multi-frame payloads.
    patch = ggml_reshape_4d(ctx, patch, pw, ph, dim, bs);
    patch = ggml_cont(ctx, ggml_permute(ctx, patch, 1, 2, 0, 3));
    patch = ggml_reshape_3d(ctx, patch, dim, patches, bs);
    g.debug_tensor("dino.patch_raw", patch);

    // DINOv2 patch embed includes its own token stack and transformer. When
    // those tensors are present, execute that path before GCT aggregation.
    if (g.find("aggregator.patch_embed.blocks.0.norm1.weight")) {
        auto *cls = g.repeated_token({"aggregator.patch_embed.cls_token"}, dim,
                                     1, bs);
        auto *dreg = g.repeated_token(
                {"aggregator.patch_embed.register_tokens"}, dim, 4, bs);
        g.debug_tensor("dino.cls", cls);
        g.debug_tensor("dino.register", dreg);
        auto *cls_patch = ggml_concat(ctx, cls, patch, 1);
        if (auto *pe = g.find("aggregator.patch_embed.pos_embed")) {
            // The exported checkpoint contains a fixed 37x37 DINO table. The
            // validated 518x518 profile adds it unchanged; any other running
            // grid resamples it on the host exactly like DINOv2's
            // interpolate_pos_encoding instead of aborting inside ggml.
            const int64_t stored_tokens = pe->ne[1];
            const int64_t m = static_cast<int64_t>(
                    std::sqrt(static_cast<double>(stored_tokens - 1)));
            if (pe->ne[0] != dim || m * m + 1 != stored_tokens) {
                p_->error = "DINO positional embedding table is malformed: " +
                            std::to_string(pe->ne[0]) + "x" +
                            std::to_string(stored_tokens);
                ggml_free(ctx);
                return false;
            }
            if (stored_tokens == 1 + patches && ph == pw) {
                auto *pos = ggml_reshape_3d(ctx, pe, dim, 1 + patches, 1);
                pos = ggml_repeat_4d(ctx, pos, dim, 1 + patches, bs, 1);
                g.debug_tensor("dino.pos", pos);
                cls_patch = ggml_add(ctx, cls_patch, g.f32(pos));
            } else if (!p_->pos_embed_host.empty()) {
                std::vector<float> table = interpolate_dino_pos_table(
                        p_->pos_embed_host, dim, static_cast<int>(m),
                        static_cast<int>(pw), static_cast<int>(ph));
                auto *pos = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim,
                                               1 + patches, 1, 1);
                g.constant_tensors.emplace_back(pos, std::move(table));
                pos = ggml_repeat_4d(ctx, pos, dim, 1 + patches, bs, 1);
                g.debug_tensor("dino.pos", pos);
                cls_patch = ggml_add(ctx, cls_patch, pos);
            } else {
                p_->error =
                        "DINO positional embedding could not be read from the "
                        "GGUF; "
                        "host interpolation requires an unquantized pos_embed "
                        "tensor";
                ggml_free(ctx);
                return false;
            }
        }
        g.debug_tensor("dino.cls_patch", cls_patch);
        auto *cls_only = ggml_view_3d(ctx, cls_patch, dim, 1, bs,
                                      cls_patch->nb[1], cls_patch->nb[2], 0);
        auto *patch_only =
                ggml_view_3d(ctx, cls_patch, dim, patches, bs, cls_patch->nb[1],
                             cls_patch->nb[2], cls_patch->nb[1]);
        auto *cp = ggml_concat(ctx, cls_only, dreg, 1);
        cp = ggml_concat(ctx, cp, patch_only, 1);
        g.debug_tensor("dino.tokens", cp);
        auto *patch_with_tokens = cp;
        g.debug_tensor("dino.input", patch_with_tokens);
        for (int i = 0; i < layers; ++i) {
            auto *z = g.block(
                    patch_with_tokens,
                    "aggregator.patch_embed.blocks." + std::to_string(i), heads,
                    false);
            if (!z) {
                p_->error = g.error;
                ggml_free(ctx);
                return false;
            }
            patch_with_tokens = z;
            g.debug_tensor("dino.block" + std::to_string(i), patch_with_tokens);
        }
        patch_with_tokens =
                g.norm(patch_with_tokens, "aggregator.patch_embed.norm");
        if (!patch_with_tokens) {
            p_->error = g.error;
            ggml_free(ctx);
            return false;
        }
        patch = ggml_view_3d(ctx, patch_with_tokens, dim, patches, bs,
                             patch_with_tokens->nb[1], patch_with_tokens->nb[2],
                             5 * patch_with_tokens->nb[1]);
        g.debug_tensor("dino.patch", patch);
    }

    auto *cam = g.repeated_token({"aggregator.camera_token"}, dim, 1, bs, true);
    auto *reg =
            g.repeated_token({"aggregator.register_token"}, dim, 4, bs, true);
    auto *scale = g.repeated_token({"aggregator.scale_token"}, dim, 1, bs, true,
                                   true);
    auto *tokens = ggml_concat(ctx, cam, reg, 1);
    tokens = ggml_concat(ctx, tokens, scale, 1);
    tokens = ggml_concat(ctx, tokens, patch, 1);
    // This is the complete per-frame state entering the frame transformer.
    // Keep it observable to distinguish stream-token/cached-state divergence
    // from an error introduced by the transformer itself.
    g.debug_tensor("frame.input", tokens);
    const int64_t special = 6;
    const int64_t per_frame = special + patches;
    std::vector<ggml_tensor *> selected;
    for (int i = 0; i < layers; ++i) {
        auto *frame_in = tokens;
        auto *frame = g.block(frame_in,
                              "aggregator.frame_blocks." + std::to_string(i),
                              heads, true);
        if (!frame) {
            p_->error = g.error;
            ggml_free(ctx);
            return false;
        }
        g.debug_tensor("frame.block" + std::to_string(i), frame);
        auto *global_in =
                ggml_reshape_3d(ctx, frame, dim, per_frame * frames, batch);
        auto *global = g.block(global_in,
                               "aggregator.global_blocks." + std::to_string(i),
                               heads, true);
        if (!global) {
            p_->error = g.error;
            ggml_free(ctx);
            return false;
        }
        g.debug_tensor("global.block" + std::to_string(i), global);
        global = ggml_reshape_3d(ctx, global, dim, per_frame, bs);
        tokens = global;
        if (i == 4 || i == 11 || i == 17 || i == 23) {
            selected.push_back(
                    ggml_cont(ctx, ggml_concat(ctx, frame, global, 0)));
        }
    }
    if (selected.empty())
        selected.push_back(ggml_concat(ctx, tokens, tokens, 0));

    // CameraCausalHead: iterative pose refinement using the final camera token.
    auto *final_features = ggml_cont(ctx, selected.back());
    // Feature dumps are a verification interface.  Capture this tensor at
    // construction time because its normal buffer is eligible for reuse once
    // the camera and DPT consumers have completed.
    ggml_tensor *final_features_dump = nullptr;
    if (dump_features) final_features_dump = ggml_dup(ctx, final_features);
    // CameraCausalHead phase 1 attends over its scale-frame sequence.  The
    // previous flat [C, B*S] view silently treated every scale frame as an
    // independent batch element, which prevented its causal KV cache from
    // matching the PyTorch scale pass.
    auto *pose_token =
            ggml_cont(ctx, ggml_view_2d(ctx, final_features, 2 * dim, bs,
                                        final_features->nb[2], 0));
    const int64_t expected_pose_elements = 2 * dim * frames * batch;
    if (ggml_nelements(pose_token) != expected_pose_elements) {
        p_->error = "CameraCausalHead pose-token shape mismatch: got " +
                    std::to_string(ggml_nelements(pose_token)) + ", expected " +
                    std::to_string(expected_pose_elements);
        ggml_free(ctx);
        return false;
    }
    pose_token = ggml_reshape_3d(ctx, pose_token, 2 * dim, frames, batch);
    pose_token = g.norm(pose_token, "camera_head.token_norm");
    if (!pose_token) {
        p_->error = g.error;
        ggml_free(ctx);
        return false;
    }
    g.debug_tensor("camera.token", pose_token);
    auto *pose = ggml_reshape_3d(
            ctx, g.repeated_token({"camera_head.empty_pose_tokens"}, 9, 1, bs),
            9, frames, batch);
    for (int iter = 0; iter < 4; ++iter) {
        auto *module = g.linear(pose, "camera_head.embed_pose");
        if (!module) {
            p_->error = g.error;
            ggml_free(ctx);
            return false;
        }
        g.debug_tensor("camera.iter" + std::to_string(iter) + ".embed", module);
        module = ggml_silu(ctx, module);
        g.debug_tensor("camera.iter" + std::to_string(iter) + ".embed_silu",
                       module);
        auto *mod = g.linear(module, "camera_head.poseLN_modulation.1");
        if (!mod) {
            p_->error = g.error;
            ggml_free(ctx);
            return false;
        }
        g.debug_tensor("camera.iter" + std::to_string(iter) + ".mod", mod);
        auto *adaln =
                g.norm(pose_token, "camera_head.adaln_norm", 1e-6f, false);
        g.debug_tensor("camera.adaln", adaln);
        auto *shift = ggml_view_2d(ctx, mod, 2 * dim, bs, mod->nb[1], 0);
        auto *scale_m = ggml_view_2d(ctx, mod, 2 * dim, bs, mod->nb[1],
                                     2 * dim * ggml_type_size(mod->type));
        auto *gate = ggml_view_2d(ctx, mod, 2 * dim, bs, mod->nb[1],
                                  4 * dim * ggml_type_size(mod->type));
        auto *one = ggml_reshape_2d(
                ctx, ggml_repeat_4d(ctx, g.scalar(1.0f), 2 * dim, bs, 1, 1),
                2 * dim, bs);
        auto *modulated = ggml_add(
                ctx, ggml_mul(ctx, adaln, ggml_add(ctx, one, scale_m)), shift);
        modulated = ggml_mul(ctx, gate, modulated);
        modulated = ggml_add(ctx, modulated, pose_token);
        g.debug_tensor("camera.iter" + std::to_string(iter) + ".input",
                       modulated);
        for (int j = 0; j < 4; ++j) {
            auto *z =
                    g.block(modulated, "camera_head.trunk." + std::to_string(j),
                            heads, false,
                            "camera_head.iter." + std::to_string(iter) +
                                    ".trunk." + std::to_string(j));
            if (!z) {
                p_->error = g.error;
                ggml_free(ctx);
                return false;
            }
            modulated = z;
            g.debug_tensor("camera.iter" + std::to_string(iter) + ".block" +
                                   std::to_string(j),
                           modulated);
        }
        auto *delta = g.norm(modulated, "camera_head.trunk_norm");
        if (!delta) {
            p_->error = g.error;
            ggml_free(ctx);
            return false;
        }
        delta = g.linear(delta, "camera_head.pose_branch.fc1");
        if (!delta) {
            p_->error = g.error;
            ggml_free(ctx);
            return false;
        }
        delta = ggml_gelu_erf(ctx, delta);
        delta = g.linear(delta, "camera_head.pose_branch.fc2");
        if (!delta) {
            p_->error = g.error;
            ggml_free(ctx);
            return false;
        }
        g.debug_tensor("camera.iter" + std::to_string(iter) + ".delta", delta);
        pose = iter == 0 ? delta : ggml_add(ctx, pose, delta);
    }

    // DPTHead: all four selected token layers, projections, resize layers,
    // scratch convolutions, RefineNet fusion, and output head.
    if (selected.size() != 4) {
        p_->error =
                "GCT checkpoint must expose selected layers [4,11,17,23] for "
                "DPT";
        ggml_free(ctx);
        return false;
    }
    std::vector<ggml_tensor *> dpt_features;
    dpt_features.reserve(4);
    for (auto *layer : selected) {
        auto *patch_features =
                ggml_view_3d(ctx, layer, 2 * dim, patches, bs, layer->nb[1],
                             layer->nb[2], special * layer->nb[1]);
        dpt_features.push_back(patch_features);
    }
    auto *depth_pair = g.dpt(dpt_features, pw, ph, 2 * dim, bs, width, height);
    if (!depth_pair) {
        p_->error = g.error;
        ggml_free(ctx);
        return false;
    }
    auto *depth = ggml_reshape_3d(
            ctx,
            ggml_cont(ctx, ggml_view_4d(ctx, depth_pair, width, height, 1, bs,
                                        depth_pair->nb[1], depth_pair->nb[2],
                                        depth_pair->nb[3], 0)),
            1, width * height, bs);
    auto *conf = ggml_reshape_3d(
            ctx,
            ggml_cont(ctx, ggml_view_4d(ctx, depth_pair, width, height, 1, bs,
                                        depth_pair->nb[1], depth_pair->nb[2],
                                        depth_pair->nb[3], depth_pair->nb[2])),
            1, width * height, bs);
    depth = ggml_exp(ctx, depth);
    conf = ggml_add(ctx, ggml_exp(ctx, conf),
                    ggml_reshape_3d(ctx,
                                    ggml_repeat_4d(ctx, g.scalar(1.0f), 1,
                                                   width * height, bs, 1),
                                    1, width * height, bs));

    auto *pose_out = ggml_cont(ctx, ggml_reshape_2d(ctx, pose, 9, bs));
    auto *fov = ggml_view_2d(ctx, pose_out, 2, bs, pose_out->nb[1],
                             7 * ggml_type_size(pose_out->type));
    // The FoV slice is strided for multi-frame batches, and CUDA's unary
    // kernels reject non-contiguous operands (bs=1 only hides it because
    // ggml skips size-1 axes). Materialize the slice so every backend
    // evaluates the identical ReLU, then write it back into the pose.
    fov = ggml_cont(ctx, fov);
    fov = ggml_relu(ctx, fov);
    pose_out = ggml_set(ctx, pose_out, fov, pose_out->nb[1], pose_out->nb[2],
                        pose_out->nb[3], 7 * ggml_type_size(pose_out->type));
    struct prof_t {
        bool on = false;
        std::chrono::steady_clock::time_point t0;
        void begin() {
            on = g_opt && g_opt->profile;
            t0 = std::chrono::steady_clock::now();
        }
        void mark(const char *what) {
            if (!on) return;
            auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "PROF %s %.0f ms\n", what,
                         std::chrono::duration<double, std::milli>(now - t0)
                                 .count());
            t0 = now;
        }
    } prof;
    prof.begin();
    auto *graph = ggml_new_graph_custom(ctx, 100000, false);
    // Pre-phase resident writes first: the attention history read below
    // consumes the special tokens they just copied into the payload.
    for (const auto &w : g.res_presize) ggml_build_forward_expand(graph, w);
    ggml_build_forward_expand(graph, pose_out);
    ggml_build_forward_expand(graph, depth);
    ggml_build_forward_expand(graph, conf);
    if (final_features_dump)
        ggml_build_forward_expand(graph, final_features_dump);
    // Keep requested diagnostic tensors live until graph completion.  Without
    // making them graph roots, gallocr may legally reuse their buffers before
    // ggml_backend_tensor_get serializes the values below.
    for (const auto &item : g.debug_tensors)
        ggml_build_forward_expand(graph, item.second);
    // Batch all cache captures into ONE flat readback node: 48 individual
    // ggml_backend_tensor_get calls each synchronize the compute stream
    // (~65 ms each, 3.2 s per frame), while a single readback of the same
    // bytes costs one synchronization plus the PCIe transfer.
    // Group the captures by (type, ne) — per-layer cache token counts differ
    // between the patch-window layers and the global layers, which keep the
    // special tokens — and build one flat concat per group. Each group is a
    // single synchronized readback instead of 2 readbacks per layer.
    struct cap_group {
        ggml_tensor *flat = nullptr;
        std::vector<const graph_builder::kv_capture *> caps;
    };
    std::vector<cap_group> capture_groups;
    const bool flat_disabled = g_opt && g_opt->flat_off;
    for (const auto &cap : g.cache_captures) {
        cap_group *grp = nullptr;
        for (auto &g2 : capture_groups) {
            const auto *first = g2.caps[0];
            if (first->k->type == cap.k->type &&
                first->k->ne[0] == cap.k->ne[0] &&
                first->k->ne[1] == cap.k->ne[1] &&
                first->k->ne[2] == cap.k->ne[2]) {
                grp = &g2;
                break;
            }
        }
        if (!grp) {
            capture_groups.push_back({});
            grp = &capture_groups.back();
        }
        grp->caps.push_back(&cap);
    }
    for (auto &grp : capture_groups) {
        if (flat_disabled) {
            grp.flat = nullptr;
            continue;
        }
        // reshape every capture to 1-D first: concatenating the raw 3-D
        // tensors along ne0 would interleave heads/tokens rows instead of
        // producing a plain per-layer byte concatenation
        for (const auto *cap : grp.caps) {
            ggml_tensor *k1 =
                    ggml_reshape_1d(ctx, cap->k, ggml_nelements(cap->k));
            grp.flat = grp.flat ? ggml_concat(ctx, grp.flat, k1, 0) : k1;
        }
        for (const auto *cap : grp.caps) {
            ggml_tensor *v1 =
                    ggml_reshape_1d(ctx, cap->v, ggml_nelements(cap->v));
            grp.flat = ggml_concat(ctx, grp.flat, v1, 0);
        }
        ggml_build_forward_expand(graph, grp.flat);
    }
    // Device-resident persistence writes: expanded after every reader so the
    // attention consumes the pre-write payload within this graph.
    for (const auto &w : g.res_writes) ggml_build_forward_expand(graph, w);
    prof.mark("graph-build");
    // Pooled allocator: reuse when the graph shape is unchanged (the common
    // steady-state streaming case), rebuild on mismatch instead of letting
    // gallocr re-reserve in place (the stale-memory defect lives exactly
    // there). See the pooled_allocr note in impl.
    const size_t shape_key = graph_shape_fingerprint(graph);
    if (!p_->pooled_allocr || p_->pooled_key != shape_key) {
        if (p_->pooled_allocr) ggml_gallocr_free(p_->pooled_allocr);
        p_->pooled_allocr = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(p_->be.get()));
        p_->pooled_key = shape_key;
    }
    auto *allocr = p_->pooled_allocr;
    if (!allocr || !ggml_gallocr_alloc_graph(allocr, graph)) {
        p_->error = "GGML backend graph allocation failed";
        if (p_->pooled_allocr) {
            ggml_gallocr_free(p_->pooled_allocr);
            p_->pooled_allocr = nullptr;
            p_->pooled_key = 0;
        }
        ggml_free(ctx);
        return false;
    }
    // Public input contract is PyTorch-style [B,F,C,H,W], while GGML's
    // 4-D tensor is contiguous as [W,H,C,B]. Repack explicitly so channel
    // and spatial order stays identical across the two implementations.
    std::vector<float> input_whcb(static_cast<size_t>(bs) * channels * height *
                                  width);
    for (int b = 0; b < batch; ++b)
        for (int f = 0; f < frames; ++f) {
            const int bf = b * frames + f;
            for (int c = 0; c < channels; ++c)
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        const size_t src = (((static_cast<size_t>(b) * frames +
                                              f) * channels +
                                             c) * height +
                                            y) * width +
                                           x;
                        const size_t dst =
                                ((static_cast<size_t>(bf) * channels + c) *
                                         height +
                                 y) * width +
                                x;
                        input_whcb[dst] = images[src];
                    }
        }
    ggml_backend_tensor_set(input, input_whcb.data(), 0,
                            input_whcb.size() * sizeof(float));
    if (mean->buffer) ggml_backend_tensor_set(mean, mean_v, 0, sizeof(mean_v));
    if (stdev->buffer) ggml_backend_tensor_set(stdev, std_v, 0, sizeof(std_v));
    for (const auto &item : g.scalar_tensors)
        if (item.first->buffer)
            ggml_backend_tensor_set(item.first, &item.second, 0, sizeof(float));
    for (const auto &item : g.constant_tensors) {
        if (item.first->buffer && !item.second.empty()) {
            ggml_backend_tensor_set(item.first, item.second.data(), 0,
                                    item.second.size() * sizeof(float));
        }
    }
    for (const auto &item : g.position_tensors) {
        auto *pos = item.first;
        if (!pos->buffer) continue;
        std::vector<int32_t> values(static_cast<size_t>(pos->ne[0]));
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = item.second + static_cast<int32_t>(i);
        ggml_backend_tensor_set(pos, values.data(), 0,
                                values.size() * sizeof(int32_t));
    }
    for (const auto &ci : g.cache_inputs) {
        if (!ci.tensor->buffer || ci.src->empty()) continue;
        if (!ci.f16) {
            ggml_backend_tensor_set(ci.tensor, ci.src->data(), 0,
                                    ci.src->size() * sizeof(float));
            continue;
        }
        // F16 mode: convert the host F32 sidecar on the fly. The conversion
        // dominated the per-frame profile (scalar ggml_fp32_to_fp16 = 60% of
        // total wall time on a 100-frame stream), so use ggml-cpu's array
        // entry point, which lowers to the F16C hardware instructions. Still
        // threaded: the steady-state cache reaches gigabytes per upload.
        const size_t n = ci.src->size();
        // Stage the F16 bytes in PINNED host memory: a pageable source forces
        // the driver to copy through an internal staging buffer (kernel-side,
        // byte by byte), which pinned DMA avoids. Profiled: the pageable path
        // dominated the whole per-frame wall time on long streams.
        static std::vector<ggml_fp16_t> f16buf_v;
        if (f16buf_v.size() < n) f16buf_v.resize(n);
        ggml_fp16_t *f16buf = f16buf_v.data();
        ggml_backend_buffer_t pin_buf = nullptr;
        const std::vector<float> &src = *ci.src;
        const int64_t total = static_cast<int64_t>(n);
        const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
        const int64_t chunk = (total + hc - 1) / hc;
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < hc && t * chunk < total; ++t) {
            workers.emplace_back([&, t]() {
                const int64_t lo = t * chunk;
                const int64_t hi = std::min(total, lo + chunk);
                for (int64_t i = lo; i < hi; ++i)
                    f16buf[i] = ggml_fp32_to_fp16(src[i]);
            });
        }
        for (auto &w : workers) w.join();
        ggml_backend_tensor_set(ci.tensor, f16buf, 0, n * sizeof(ggml_fp16_t));
    }
    prof.mark("cache-upload");
    if (ggml_backend_graph_compute(p_->be.get(), graph) !=
        GGML_STATUS_SUCCESS) {
        p_->error = "GGML GCT graph execution failed";
        if (p_->pooled_allocr) {
            ggml_gallocr_free(p_->pooled_allocr);
            p_->pooled_allocr = nullptr;
            p_->pooled_key = 0;
        }
        ggml_free(ctx);
        return false;
    }
    prof.mark("graph-compute");
    result->batch = batch;
    result->frames = frames;
    result->height = height;
    result->width = width;
    result->pose_enc.resize(static_cast<size_t>(bs) * 9);
    result->depth.resize(static_cast<size_t>(bs) * height * width);
    result->depth_conf.resize(static_cast<size_t>(bs) * height * width);
    ggml_backend_tensor_get(pose_out, result->pose_enc.data(), 0,
                            result->pose_enc.size() * sizeof(float));
    ggml_backend_tensor_get(depth, result->depth.data(), 0,
                            result->depth.size() * sizeof(float));
    ggml_backend_tensor_get(conf, result->depth_conf.data(), 0,
                            result->depth_conf.size() * sizeof(float));
    result->c2w.resize(static_cast<size_t>(bs) * 16);
    result->intrinsics.resize(static_cast<size_t>(bs) * 4);
    for (int64_t i = 0; i < bs; ++i) {
        const float *pose = result->pose_enc.data() + i * 9;
        const float x = pose[3], y = pose[4], z = pose[5], r = pose[6];
        const float two_s = 2.0f / (x * x + y * y + z * z + r * r);
        // pose_enc stores the camera-from-world (w2c) transform [R|t], matching
        // pose_encoding_to_extri_intri. Consumers (viewer unprojection, BSS
        // artifacts) expect camera-to-world, so emit the proper inverse:
        // c2w_rot = R^T and c2w_t = -R^T t. Copying [R|t] verbatim rotated
        // every reconstructed frame by the w2c/c2w discrepancy.
        const float r00 = 1.0f - two_s * (y * y + z * z);
        const float r01 = two_s * (x * y - z * r);
        const float r02 = two_s * (x * z + y * r);
        const float r10 = two_s * (x * y + z * r);
        const float r11 = 1.0f - two_s * (x * x + z * z);
        const float r12 = two_s * (y * z - x * r);
        const float r20 = two_s * (x * z - y * r);
        const float r21 = two_s * (y * z + x * r);
        const float r22 = 1.0f - two_s * (x * x + y * y);
        const float tx = pose[0], ty = pose[1], tz = pose[2];
        float *c2w = result->c2w.data() + i * 16;
        c2w[0] = r00;
        c2w[1] = r10;
        c2w[2] = r20;
        c2w[3] = -(r00 * tx + r10 * ty + r20 * tz);
        c2w[4] = r01;
        c2w[5] = r11;
        c2w[6] = r21;
        c2w[7] = -(r01 * tx + r11 * ty + r21 * tz);
        c2w[8] = r02;
        c2w[9] = r12;
        c2w[10] = r22;
        c2w[11] = -(r02 * tx + r12 * ty + r22 * tz);
        c2w[12] = 0.0f;
        c2w[13] = 0.0f;
        c2w[14] = 0.0f;
        c2w[15] = 1.0f;
        float *intrinsics = result->intrinsics.data() + i * 4;
        intrinsics[0] = (width / 2.0f) / std::tan(pose[8] / 2.0f);
        intrinsics[1] = (height / 2.0f) / std::tan(pose[7] / 2.0f);
        intrinsics[2] = width / 2.0f;
        intrinsics[3] = height / 2.0f;
    }
    if (dump_features)
        if (const char *dump = g_opt->dump_features_dir.c_str()) {
            const size_t n = static_cast<size_t>(
                    final_features_dump->ne[0] * final_features_dump->ne[1] *
                    final_features_dump->ne[2] * final_features_dump->ne[3]);
            std::vector<float> values(n);
            ggml_backend_tensor_get(final_features_dump, values.data(), 0,
                                    n * sizeof(float));
            std::string dump_path = dump;
            if (frames == 1 && p_->backend_name != "cpu")
                dump_path += "." + std::to_string(inference_index);
            std::ofstream out(dump_path, std::ios::binary);
            out.write(reinterpret_cast<const char *>(values.data()),
                      static_cast<std::streamsize>(n * sizeof(float)));
        }
    if (dump_stages || dump_camera_stages || g.dump_internal)
        if (const char *dump = (dump_stages || g.dump_internal
                                        ? g_opt->dump_stages_dir
                                        : g_opt->dump_camera_stages_dir)
                                       .c_str()) {
            std::fprintf(stderr,
                         "DBG dump write: dump_stages=%d dump_internal=%d "
                         "debug_tensors=%zu path=%s\n",
                         (int)dump_stages, (int)g.dump_internal,
                         g.debug_tensors.size(), dump);
            for (size_t di = 0; di < g.debug_tensors.size(); ++di) {
                const auto &item = g.debug_tensors[di];
                const auto *tensor = item.second;
                const size_t n =
                        static_cast<size_t>(tensor->ne[0] * tensor->ne[1] *
                                            tensor->ne[2] * tensor->ne[3]);
                std::vector<float> values(n);
                ggml_backend_tensor_get(tensor, values.data(), 0,
                                        n * sizeof(float));
                std::string path = std::string(dump) + "." + item.first;
                if (frames == 1 && p_->backend_name != "cpu")
                    path += "." + std::to_string(inference_index);
                if (di < 2)
                    std::fprintf(stderr, "DBG writing [%zu] %s (n=%zu)\n", di,
                                 path.c_str(), n);
                std::ofstream out(path, std::ios::binary);
                if (!out) {
                    std::fprintf(stderr, "DBG open FAILED: %s\n", path.c_str());
                    continue;
                }
                out.write(reinterpret_cast<const char *>(values.data()),
                          static_cast<std::streamsize>(n * sizeof(float)));
            }
        }
    if (!capture_groups.empty()) {
        // one synchronized readback per shape group, then split host-side
        for (auto &grp : capture_groups) {
            if (!grp.flat || grp.flat->type != GGML_TYPE_F32)
                continue;  // flash-mode F16 path falls back below
            const size_t total = (size_t)ggml_nelements(grp.flat);
            // D2H readback through PINNED memory: a pageable destination
            // makes cudaMemcpy stage through the driver kernel-side, which
            // measured ~3 s per frame for this single buffer.
            static ggml_backend_buffer_t read_buf = nullptr;
            static size_t read_cap = 0;
            if (read_cap < total) {
                if (read_buf) ggml_backend_buffer_free(read_buf);
                read_cap = total + total / 4;
                ggml_backend_buffer_type_t host_buft =
                        ggml_backend_dev_host_buffer_type(
                                ggml_backend_get_device(p_->be.get()));
                read_buf = ggml_backend_buft_alloc_buffer(
                        host_buft ? host_buft
                                  : ggml_backend_get_default_buffer_type(
                                            p_->be.get()),
                        read_cap * sizeof(float));
            }
            float *flat =
                    read_buf ? (float *)ggml_backend_buffer_get_base(read_buf)
                             : (float *)std::calloc(total, sizeof(float));
            ggml_backend_tensor_get(grp.flat, flat, 0, total * sizeof(float));
            prof.mark("  capture:get");
            // the flat layout is ALL K slices (graph order) followed by ALL
            // V slices, matching the two concat loops that built it
            const size_t n_caps = grp.caps.size();
            const size_t n0 = (size_t)grp.caps[0]->tokens *
                              grp.caps[0]->k->ne[0] * grp.caps[0]->k->ne[1];
            for (size_t i = 0; i < n_caps; ++i) {
                const auto &cap = *grp.caps[i];
                auto &state = p_->kv_cache[cap.key];
                const size_t n =
                        (size_t)cap.tokens * cap.k->ne[0] * cap.k->ne[1];
                std::vector<float> frame_k(flat + i * n, flat + (i + 1) * n);
                std::vector<float> frame_v(flat + n_caps * n0 + i * n,
                                           flat + n_caps * n0 + (i + 1) * n);
                append_cache_capture(&state, cap, frame_k, frame_v,
                                     g.cache_scale_frames,
                                     g.cache_window_frames);
            }
            prof.mark("  capture:append");
        }
    } else
        for (const auto &cap : g.cache_captures) {
            auto &state = p_->kv_cache[cap.key];
            const size_t n = static_cast<size_t>(cap.tokens * cap.k->ne[0] *
                                                 cap.k->ne[1]);
            std::vector<float> frame_k(n), frame_v(n);
            if (cap.k->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> k16(n), v16(n);
                ggml_backend_tensor_get(cap.k, k16.data(), 0,
                                        n * sizeof(ggml_fp16_t));
                ggml_backend_tensor_get(cap.v, v16.data(), 0,
                                        n * sizeof(ggml_fp16_t));
                for (size_t i = 0; i < n; ++i) {
                    frame_k[i] = ggml_fp16_to_fp32(k16[i]);
                    frame_v[i] = ggml_fp16_to_fp32(v16[i]);
                }
            } else {
                ggml_backend_tensor_get(cap.k, frame_k.data(), 0,
                                        n * sizeof(float));
                ggml_backend_tensor_get(cap.v, frame_v.data(), 0,
                                        n * sizeof(float));
            }
            append_cache_capture(&state, cap, frame_k, frame_v,
                                 g.cache_scale_frames, g.cache_window_frames);
        }
    // keep the pooled gallocr for the next frame's same-shape reuse
    prof.mark("cache-capture");
    ggml_free(ctx);
    return true;
}

const model_info &model::info() const { return p_->info; }

const std::string &model::device_name() const { return p_->be.device_name(); }

ggml_backend_t model::backend_handle() const { return p_->be.get(); }

int model::threads() const { return p_->threads_resolved; }
const std::string &model::error() const { return p_->error; }

}  // namespace lingbot
