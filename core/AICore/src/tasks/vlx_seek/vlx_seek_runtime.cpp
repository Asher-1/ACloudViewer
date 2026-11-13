// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/vlx_seek/vlx_seek_runtime.h"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <regex>
#include <thread>
#include <utility>

#include "common/aicore_log.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "tasks/vlx_seek/third_party/stb/stb_image.h"
#include "tasks/vlx_seek/third_party/stb/stb_image_write.h"
#include "tasks/vlx_seek/vlx_ggml_util.h"
#include "tasks/vlx_seek/vlx_gguf_loader.h"
#include "tasks/vlx_seek/vlx_hfre.h"
#include "tasks/vlx_seek/vlx_image_prep.h"
#include "tasks/vlx_seek/vlx_mmproj_graph.h"
#include "tasks/vlx_seek/vlx_radio_graph.h"
#include "tasks/vlx_seek/vlx_simplefpn_graph.h"
#include "tasks/vlx_seek/vlx_wedetect_graph.h"

#define VLX_LOG_DEBUG(...) ::vlx::logf(AICORE_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define VLX_LOG_INFO(...) ::vlx::logf(AICORE_LOG_LEVEL_INFO, __VA_ARGS__)
#define VLX_LOG_WARN(...) ::vlx::logf(AICORE_LOG_LEVEL_WARN, __VA_ARGS__)
#define VLX_LOG_ERROR(...) ::vlx::logf(AICORE_LOG_LEVEL_ERROR, __VA_ARGS__)

namespace vlx {

void logf(int level, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buf[1024];
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    // aicore_log_at(level, tag, fmt, ...) concatenates the literal tag and
    // formats fmt separately; passing the tagged string as fmt lost the
    // message and printed a literal "%s" into the session log.
    aicore_log_at(level, "[AICore vlx_seek] ", "%s\n", buf);
}

void set_log_level(int level) { aicore_set_log_level(level); }

namespace {

constexpr const char* kTasks[] = {
        "vqa",       "detection",           "grounding_single",
        "counting",  "reasoning_detection", "brief_region_caption",
        "region_ocr"};

bool known_task(const std::string& task) {
    return std::find_if(std::begin(kTasks), std::end(kTasks),
                        [&](const char* t) { return task == t; }) !=
           std::end(kTasks);
}

}  // namespace

struct VlxSeekRuntime::Impl {
    VlxSeekOptions opts;
    int threads = 0;
    std::string device_resolved;

    // Physical backend via the shared lease registry (yolo-task pattern).
    aicore::runtime::BackendLease gpu_lease, cpu_lease;
    ggml_backend_t backend = nullptr;  // lease handle; freed by the lease

    // models (GGUF bytes stay resident for lazy graph (re)builds)
    GgufFile wedetect_gguf, vision_gguf;
    // lazy, RESIDENT graph objects: weights upload once, every run reuses
    // them (destroying the objects would free the device weights)
    std::unique_ptr<WeDetectGraph> wedetect;
    std::unique_ptr<MmprojGraph> mm;
    std::unique_ptr<SimpleFpnGraph> fpn;
    std::unique_ptr<RadioGraph> radio;
    std::unique_ptr<HfreEncoder> hfre;

    void dump(const char* name, const void* p, size_t bytes) const {
        if (!opts.dump_dir.empty())
            write_file(std::string(opts.dump_dir) + "/" + name, p, bytes);
    }

    llama_model* llm = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_context* lctx = nullptr;

    // last run state (exposed by the result getters / info_json)
    bool has_result = false;
    std::string answer, result_json;
    std::vector<ResultBox> boxes;
    int32_t prompt_tokens = 0, completion_tokens = 0, proposal_count = 0;
    double preprocess_ms = 0, vision_ms = 0, llm_ms = 0, postprocess_ms = 0,
           serialization_ms = 0, e2e_ms = 0;
    std::string last_error;
    bool ready = false;

    std::string derive(const char* stem, const char* prec) const {
        return opts.models_dir + "/" + stem + "-" + prec + ".gguf";
    }
};

VlxSeekRuntime::VlxSeekRuntime(const VlxSeekOptions& opts) : impl_(new Impl) {
    impl_->opts = opts;
    impl_->threads = opts.threads > 0 ? opts.threads
                                      : int(ggml_common::default_cpu_threads());
}

VlxSeekRuntime::~VlxSeekRuntime() {
    if (impl_->lctx) llama_free(impl_->lctx);
    if (impl_->llm) llama_model_free(impl_->llm);
    impl_->gpu_lease.reset();
    impl_->cpu_lease.reset();
    delete impl_;
}

bool VlxSeekRuntime::load() {
    Impl& s = *impl_;
    if (s.ready) return true;
    aicore_set_log_level(s.opts.log_level);

    ggml_common::load_backends_once();

    // ---- device resolution (shared runtime registry, yolo-task pattern) ----
    const std::string want = ggml_common::to_lower(
            s.opts.device.empty() ? "auto" : s.opts.device);
    const bool force_cpu = want == "cpu";
    if (!force_cpu) {
        ggml_common::GpuBackendGroup group =
                ggml_common::resolve_gpu_group(want);
        if (group.primary()) {
            s.gpu_lease = aicore::runtime::adopt_backend_lease(
                    group.gpus[0], group.names[0], s.threads);
            group.gpus[0] = nullptr;
            group.release();
            if (s.gpu_lease) {
                s.backend = s.gpu_lease.handle();
                s.device_resolved = s.gpu_lease.device();
            }
        } else if (want != "auto") {
            s.last_error = "no ggml device for device '" + s.opts.device + "'";
            return false;
        }
    }
    if (!s.backend) {
        std::string cpu_error;
        s.cpu_lease = aicore::runtime::acquire_backend_lease("cpu", s.threads,
                                                             &cpu_error);
        if (!s.cpu_lease) {
            s.last_error =
                    cpu_error.empty() ? "CPU backend unavailable" : cpu_error;
            return false;
        }
        s.backend = s.cpu_lease.handle();
        s.device_resolved = s.cpu_lease.device();
        if (want != "auto" && !force_cpu)
            VLX_LOG_WARN("requested device %s not found; using CPU",
                         s.opts.device.c_str());
    }

    // ---- weight paths ----
    std::string wedetect_path =
            s.opts.wedetect_gguf.empty()
                    ? s.derive("wedetect-base-uni",
                               "f32")  // official .pth dtype
                    : s.opts.wedetect_gguf;
    std::string vision_path =
            s.opts.vision_gguf.empty()
                    ? s.derive("vlx-vision-10b", s.opts.precision.c_str())
                    : s.opts.vision_gguf;
    std::string llm_path =
            s.opts.llm_gguf.empty()
                    ? s.derive("vlx-10b-llm",
                               s.opts.precision == "f32"
                                       ? "f16"
                                       : s.opts.precision.c_str())
                    : s.opts.llm_gguf;

    // ---- GPU admission (-3 resource contract) ----
    auto file_bytes = [](const std::string& p) -> size_t {
        std::error_code ec;
        const auto sz =
                std::filesystem::file_size(std::filesystem::u8path(p), ec);
        return ec ? 0 : size_t(sz);
    };
    const size_t need = file_bytes(wedetect_path) + file_bytes(vision_path) +
                        file_bytes(llm_path);
    std::string admission_error;
    if (need > 0 && !s.cpu_lease &&  // admission only applies to accelerators
        !ggml_common::gpu_admission_check(s.opts.device, need,
                                          &admission_error)) {
        s.last_error = admission_error;
        return false;
    }

    // ---- llama/ggml logging follows the shared bridge ----
    llama_log_set(ggml_common::log_bridge_callback(), nullptr);

    // vision + detector GGUF bytes stay on the host; weights upload lazily on
    // the first run that needs them (f16 total device footprint ~20.5 GiB
    // with the LLM resident — q8_0 halves it).
    if (!s.wedetect_gguf.load(wedetect_path, &s.last_error)) {
        s.last_error = "wedetect load failed: " + s.last_error;
        return false;
    }
    if (!s.vision_gguf.load(vision_path, &s.last_error)) {
        s.last_error = "vision load failed: " + s.last_error;
        return false;
    }

    // ---- LLM: same resolved device, single-GPU (llm-task pattern) ----
    llama_model_params mp = llama_model_default_params();
    // devices = {nullptr} means "no accelerator" (CPU-only model AND context
    // buffers); devices = nullptr would mean "no restriction" and let llama
    // place op buffers on an unrelated GPU even for CPU requests.
    ggml_backend_dev_t devices[2] = {
            force_cpu ? nullptr : ggml_backend_get_device(s.backend), nullptr};
    mp.devices = devices;
    mp.n_gpu_layers = force_cpu ? 0 : -1;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    s.llm = llama_model_load_from_file(llm_path.c_str(), mp);
    if (!s.llm) {
        s.last_error = "llm load failed: " + llm_path;
        return false;
    }
    s.vocab = llama_model_get_vocab(s.llm);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 2048;
    cp.n_threads = s.threads;
    cp.n_threads_batch = s.threads;
    s.lctx = llama_init_from_model(s.llm, cp);
    if (!s.lctx) {
        s.last_error = "llama context init failed";
        return false;
    }
    VLX_LOG_INFO("vlx_seek ready on %s (precision %s)",
                 s.device_resolved.c_str(), s.opts.precision.c_str());
    s.ready = true;
    return true;
}

bool VlxSeekRuntime::ready() const { return impl_->ready; }
const std::string& VlxSeekRuntime::last_error() const {
    return impl_->last_error;
}
const std::string& VlxSeekRuntime::device() const {
    return impl_->device_resolved;
}
int VlxSeekRuntime::threads() const { return impl_->threads; }
const std::string& VlxSeekRuntime::precision() const {
    return impl_->opts.precision;
}

VlxRunResult VlxSeekRuntime::run(const VlxRunRequest& req) {
    Impl& s = *impl_;
    VlxRunResult out;
    const auto t_e2e = Clock::now();
    if (!s.ready) {
        out.code = -2;
        out.error = s.last_error.empty() ? "context not loaded" : s.last_error;
        return out;
    }
    if (!req.rgb || req.width <= 0 || req.height <= 0) {
        out.code = -1;
        out.error = "image must be non-NULL with positive dimensions";
        return out;
    }
    if (!known_task(req.task)) {
        out.code = -1;
        out.error = "unknown task: " + req.task;
        return out;
    }
    if (req.lang != "en" && req.lang != "zh") {
        out.code = -1;
        out.error = "invalid lang (en|zh)";
        return out;
    }
    const bool region_task =
            req.task == "brief_region_caption" || req.task == "region_ocr";
    if (region_task && (!req.target_indexes || req.n_targets == 0)) {
        out.code = -1;
        out.error = "a region task requires target-region-indexes";
        return out;
    }

    ImageU8 img;
    img.width = req.width;
    img.height = req.height;
    img.rgb.assign(req.rgb, req.rgb + size_t(req.width) * req.height * 3);

    // ---- proposals (inference.py: every task except vqa) ----
    const auto t_pre = Clock::now();
    std::vector<BoxD> caller_boxes;
    bool have_boxes = false;
    if (req.task != "vqa") {
        std::vector<std::vector<double>> raw;
        if (req.boxes) {
            for (int32_t i = 0; i < req.n_boxes; ++i)
                raw.push_back({req.boxes[i * 4 + 0], req.boxes[i * 4 + 1],
                               req.boxes[i * 4 + 2], req.boxes[i * 4 + 3]});
            VLX_LOG_INFO("Using %zu proposals from the caller", raw.size());
        } else {
            // WeDetect-Uni conv graph (lazy weight upload, then resident).
            if (!s.wedetect)
                s.wedetect.reset(
                        new WeDetectGraph(s.backend, &s.wedetect_gguf));
            WeDetectResult res;
            if (!s.wedetect->run(img, &res, false, &out.error)) {
                out.code = -2;
                return out;
            }
            for (const auto& b : res.boxes)
                raw.push_back({b.x1, b.y1, b.x2, b.y2});
            out.proposal_count = int32_t(res.boxes.size());
        }
        if (!validate_boxes(raw, img.width, img.height, &caller_boxes,
                            &out.error)) {
            out.code = -1;
            return out;
        }
        have_boxes = true;
    }
    out.proposal_count = int32_t(caller_boxes.size());
    s.proposal_count = out.proposal_count;
    out.preprocess_ms = ms_since(t_pre);

    // ---- question (task template + caller -> sorted object index remap) ----
    std::string question;
    if (req.task == "vqa") {
        question = req.text;
    } else {
        std::string slot = req.text;
        if (region_task) {
            std::string refs;
            for (int32_t i = 0; i < req.n_targets; ++i) {
                const int32_t idx = req.target_indexes[i];
                if (idx < 0 || size_t(idx) >= caller_boxes.size()) {
                    out.code = -1;
                    out.error = "target-region-indexes outside bbox_list";
                    return out;
                }
                refs += "<obj" + std::to_string(idx) + ">";
            }
            slot = refs;
        }
        if (!build_task_prompt(req.task, req.lang, slot, &question)) {
            out.code = -1;
            out.error =
                    "no template for task " + req.task + " / lang " + req.lang;
            return out;
        }
    }
    const std::vector<int> order = raster_order(caller_boxes);
    std::map<long long, long long> to_sorted, to_caller;
    for (size_t si = 0; si < order.size(); ++si) {
        to_sorted[order[si]] = long(si);
        to_caller[long(si)] = order[si];
    }
    question = remap_obj_tokens(question, to_sorted);
    std::vector<Box> boxes(order.size());
    for (size_t si = 0; si < order.size(); ++si) {
        const BoxD& b = caller_boxes[order[si]];
        boxes[si] = {float(b[0]), float(b[1]), float(b[2]), float(b[3])};
    }
    const size_t Nb = boxes.size();
    (void)have_boxes;

    // ---- vision (single backend, host RoIAlign: the functionally identical
    // path for every backend; the upstream fused-CUDA kernel is a pure
    // optimization that is not ported) ----
    const auto t_vis = Clock::now();
    int gh = 0, gw = 0;
    std::vector<float> pix_f32, pix_bf16, image_embeds, object_feats;
    primary_pixels(img, &gh, &gw, &pix_f32, &pix_bf16);
    s.dump("primary_pixels.f32", pix_f32.data(), pix_f32.size() * 4);
    std::vector<float> ll_tokens;
    for (int warm = 0; warm <= req.warm_runs; ++warm) {
        const bool last_pass = warm == req.warm_runs;
        const auto t_pass = Clock::now();
        {
            GgufFile& vg = s.vision_gguf;
            if (!s.mm) {
                MmprojConfig mc;
                mc.patch_size =
                        int(vg.i32("vlx.mmproj.patch_size", mc.patch_size));
                mc.temporal = int(vg.i32("vlx.mmproj.temporal", mc.temporal));
                mc.embed_dim =
                        int(vg.i32("vlx.mmproj.embed_dim", mc.embed_dim));
                mc.depth = int(vg.i32("vlx.mmproj.depth", mc.depth));
                mc.num_heads =
                        int(vg.i32("vlx.mmproj.num_heads", mc.num_heads));
                mc.mlp_hidden =
                        int(vg.i32("vlx.mmproj.mlp_hidden", mc.mlp_hidden));
                mc.hidden_out =
                        int(vg.i32("vlx.mmproj.hidden_out", mc.hidden_out));
                mc.merge = int(vg.i32("vlx.mmproj.merge", mc.merge));
                mc.pos_side = int(vg.i32("vlx.mmproj.pos_side", mc.pos_side));
                mc.rope_theta =
                        float(vg.f32v("vlx.mmproj.rope_theta", mc.rope_theta));
                s.mm.reset(new MmprojGraph(s.backend, mc, &vg));
            }
            if (!s.mm->run(pix_bf16.data(), gh, gw, &ll_tokens, nullptr,
                           &out.error)) {
                out.code = -2;
                out.error = "mmproj: " + out.error;
                return out;
            }
            image_embeds.resize(ll_tokens.size());
            for (size_t i = 0; i < ll_tokens.size(); ++i)
                image_embeds[i] = bf16_round(ll_tokens[i]);
            if (Nb) {
                SimpleFpnConfig fc;
                fc.vt_dim = int(s.vision_gguf.i32("vlx.fpn.vt_dim", 4096));
                fc.out_channels =
                        int(s.vision_gguf.i32("vlx.fpn.out_channels", 512));
                HfreConfig hc;
                fc.roi_output_size = hc.roi_output_size =
                        int(s.vision_gguf.i32("vlx.hfre.roi_output_size", 14));
                fc.roi_aligned = hc.roi_aligned =
                        s.vision_gguf.i32("vlx.hfre.roi_aligned", 1) != 0;
                hc.n_threads = s.threads;
                if (!s.fpn)
                    s.fpn.reset(
                            new SimpleFpnGraph(s.backend, fc, &s.vision_gguf));
                std::vector<FeatureLevel> vt_levels;
                if (!s.fpn->run(ll_tokens.data(), gh, gw, &vt_levels,
                                nullptr)) {
                    out.code = -2;
                    out.error = "fpn run failed";
                    return out;
                }
                int aH = 0, aW = 0;
                std::vector<float> aux_chw, aux_hwc;
                aux_pixels(img, &aH, &aW, &aux_chw, &aux_hwc);
                std::vector<FeatureLevel> aux_levels;
                {
                    RadioConfig rc;
                    rc.patch_size = int(s.vision_gguf.i32(
                            "vlx.radio.patch_size", rc.patch_size));
                    rc.embed_dim = int(s.vision_gguf.i32("vlx.radio.embed_dim",
                                                         rc.embed_dim));
                    rc.depth =
                            int(s.vision_gguf.i32("vlx.radio.depth", rc.depth));
                    rc.num_heads = int(s.vision_gguf.i32("vlx.radio.num_heads",
                                                         rc.num_heads));
                    rc.num_cls_tokens = int(s.vision_gguf.i32(
                            "vlx.radio.num_cls_tokens", rc.num_cls_tokens));
                    rc.num_registers = int(s.vision_gguf.i32(
                            "vlx.radio.num_registers", rc.num_registers));
                    for (int i = 0; i < 4; ++i)
                        rc.out_indices[i] =
                                i == 3 ? rc.depth - 1
                                       : int(std::lround(float(i + 1) *
                                                         rc.depth / 4.f)) -
                                                 1;
                    hc.aux_spatial_scale = float(s.vision_gguf.f32v(
                            "vlx.hfre.aux_spatial_scale", 1.0 / rc.patch_size));
                    if (!s.radio)
                        s.radio.reset(
                                new RadioGraph(s.backend, rc, &s.vision_gguf));
                    if (!s.radio->run(aux_hwc.data(), aH, aW, &aux_levels,
                                      &out.error)) {
                        out.code = -2;
                        out.error = "radio: " + out.error;
                        return out;
                    }
                }
                const int hb = gh * 16, wb = gw * 16;
                const float sw = float(double(wb) / img.width),
                            sh = float(double(hb) / img.height);
                std::vector<Box> vt_boxes(Nb);
                for (size_t n = 0; n < Nb; ++n)
                    vt_boxes[n] = {boxes[n].x1 * sw, boxes[n].y1 * sh,
                                   boxes[n].x2 * sw, boxes[n].y2 * sh};
                if (!s.hfre) s.hfre.reset(new HfreEncoder(hc));
                HfreEncoder& hfre = *s.hfre;
                std::vector<float> aux_nxc, vt_nxc, fused;
                if (!hfre.encode_aux(boxes, aux_levels, &aux_nxc, &out.error) ||
                    !hfre.encode_vt(vt_boxes, vt_levels, {4.f, 8.f, 16.f, 32.f},
                                    &vt_nxc, &out.error)) {
                    out.code = -2;
                    out.error = "hfre: " + out.error;
                    return out;
                }
                std::vector<FeatureLevel> vt_grid_ref(1);
                vt_grid_ref[0].width =
                        (gw / 2) * int(1.f / hc.vt_spatial_scale);
                vt_grid_ref[0].height =
                        (gh / 2) * int(1.f / hc.vt_spatial_scale);
                if (!hfre.fuse(aux_nxc, vt_nxc, vt_boxes,
                               int(aux_nxc.size() / Nb),
                               int(vt_nxc.size() / Nb), vt_grid_ref, &fused,
                               &out.error)) {
                    out.code = -2;
                    out.error = "hfre fuse: " + out.error;
                    return out;
                }
                for (float& v : fused) v = bf16_round(v);
                s.dump("hfre_fused.f32", fused.data(), fused.size() * 4);
                if (!run_mlp2x_gelu(s.backend, s.vision_gguf, "aux_proj", fused,
                                    int64_t(Nb), &object_feats)) {
                    out.code = -2;
                    out.error = "aux_proj projector failed";
                    return out;
                }
                for (float& v : object_feats) v = bf16_round(v);
            }
        }
        if (last_pass) out.vision_ms = ms_since(t_pass);
    }  // warm loop
    s.dump("image_embeds.f32", image_embeds.data(), image_embeds.size() * 4);
    if (Nb)
        s.dump("object_feats.f32", object_feats.data(),
               object_feats.size() * 4);

    // ---- prompt + LLM ----
    const int64_t n_embd = llama_model_n_embd(s.llm);
    std::string prompt_img = "<|vision_start|><image><|vision_end|>";
    if (Nb) {
        prompt_img += "\n";
        for (size_t i = 0; i < Nb; ++i)
            prompt_img += "<obj" + std::to_string(i) + "><objfeat>";
    }
    const std::string prompt = "<|im_start|>user\n" + prompt_img + "\n" +
                               question + "<|im_end|>\n<|im_start|>assistant\n";
    const size_t n_img = image_embeds.size() / size_t(n_embd);
    std::vector<Entry> entries;
    size_t img_seen = 0, obj_seen = 0;
    auto split = [](const std::string& str, const std::string& sep) {
        std::vector<std::string> parts;
        size_t p = 0, q;
        while ((q = str.find(sep, p)) != std::string::npos) {
            parts.push_back(str.substr(p, q - p));
            p = q + sep.size();
        }
        parts.push_back(str.substr(p));
        return parts;
    };
    const auto img_chunks = split(prompt, "<image>");
    for (size_t ci = 0; ci < img_chunks.size(); ++ci) {
        const auto obj_chunks = Nb ? split(img_chunks[ci], "<objfeat>")
                                   : std::vector<std::string>{img_chunks[ci]};
        for (size_t oi = 0; oi < obj_chunks.size(); ++oi) {
            if (oi > 0) {
                if (obj_seen >= Nb) {
                    out.code = -2;
                    out.error = "more <objfeat> than boxes";
                    return out;
                }
                entries.push_back({LLAMA_TOKEN_NULL,
                                   object_feats.data() + obj_seen++ * n_embd});
            }
            for (llama_token t : tokenize(s.vocab, obj_chunks[oi]))
                entries.push_back({t, nullptr});
        }
        if (ci + 1 < img_chunks.size()) {
            if (img_seen++) {
                out.code = -2;
                out.error = "more <image> tokens than input images";
                return out;
            }
            for (size_t r = 0; r < n_img; ++r)
                entries.push_back(
                        {LLAMA_TOKEN_NULL, image_embeds.data() + r * n_embd});
        }
    }
    out.prompt_tokens = int32_t(entries.size());
    if (!s.opts.dump_dir.empty()) {
        std::vector<int32_t> ids;
        for (const auto& e : entries)
            ids.push_back(e.id == LLAMA_TOKEN_NULL ? -1 : int32_t(e.id));
        s.dump("input_ids.i32", ids.data(), ids.size() * 4);
    }
    const auto t_llm = Clock::now();
    // Each request is stateless (llm-task contract): clear the KV memory so
    // results never depend on a previous request and positions restart at 0.
    llama_memory_clear(llama_get_memory(s.lctx), true);
    int64_t pos = 0;
    if (decode_entries(s.lctx, entries, n_embd, &pos)) {
        out.code = -2;
        out.error = "prefill failed";
        return out;
    }
    llama_synchronize(s.lctx);
    std::vector<llama_token> stop = tokenize(s.vocab, "<|im_end|>");
    if (stop.size() != 1) {
        out.code = -2;
        out.error = "<|im_end|> is not a single token";
        return out;
    }
    stop.push_back(llama_vocab_eos(s.vocab));
    double decode_ms = 0;
    const std::vector<llama_token> gen =
            generate(s.lctx, s.vocab, s.opts.max_new_tokens, stop, 0.f, 1.f,
                     &pos, &decode_ms);
    out.completion_tokens = int32_t(gen.size());
    out.llm_ms = ms_since(t_llm);
    if (!s.opts.dump_dir.empty()) {
        const int n_vocab = llama_vocab_n_tokens(s.vocab);
        s.dump("step0_logits.f32", llama_get_logits_ith(s.lctx, -1),
               size_t(n_vocab) * 4);
        std::string sj = "[";
        for (size_t i = 0; i < gen.size(); ++i)
            sj += (i ? ", " : "") + std::to_string(gen[i]);
        sj += "]";
        s.dump("step_argmax.json", sj.data(), sj.size());
    }
    (void)decode_ms;

    // ---- answer + result parsing ----
    const auto t_post = Clock::now();
    std::string answer;
    for (llama_token t : gen) answer += token_piece(s.vocab, t);
    for (size_t p; (p = answer.find("<|im_end|>")) != std::string::npos;)
        answer.erase(p, 10);
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    answer.erase(answer.begin(),
                 std::find_if(answer.begin(), answer.end(), not_space));
    answer.erase(std::find_if(answer.rbegin(), answer.rend(), not_space).base(),
                 answer.end());
    out.answer = remap_obj_tokens(answer, to_caller);

    if (!caller_boxes.empty()) {
        static const std::regex grp(
                R"(<ground>([\s\S]*?)</ground>\s*<objects>([\s\S]*?)</objects>)");
        static const std::regex obj(R"(<obj(\d+)>)");
        for (std::sregex_iterator it(out.answer.begin(), out.answer.end(), grp),
             end;
             it != end; ++it) {
            std::string label = (*it)[1].str();
            label.erase(label.begin(),
                        std::find_if(label.begin(), label.end(), not_space));
            label.erase(std::find_if(label.rbegin(), label.rend(), not_space)
                                .base(),
                        label.end());
            const std::string objs = (*it)[2].str();
            for (std::sregex_iterator jt(objs.begin(), objs.end(), obj);
                 jt != end; ++jt) {
                const std::string digits = (*jt)[1].str();
                if (digits.size() > 18) continue;
                const long long k = std::stoll(digits);
                if (k >= (long long)caller_boxes.size()) continue;
                const BoxD& b = caller_boxes[size_t(k)];
                out.boxes.push_back(
                        {(*jt)[0].str(), label, {b[0], b[1], b[2], b[3]}});
            }
        }
    }
    out.postprocess_ms = ms_since(t_post);
    out.inference_ms = out.vision_ms + out.llm_ms;
    out.e2e_ms = ms_since(t_e2e);

    // official JSON (compatibility/export boundary; the typed getters drive
    // frame inference)
    std::string js = "{\n  \"answer\": " + py_json_str(out.answer) +
                     ",\n  \"result_bbox_list\": ";
    if (out.boxes.empty()) {
        js += "[]";
    } else {
        js += "[\n";
        for (size_t i = 0; i < out.boxes.size(); ++i) {
            const auto& r = out.boxes[i];
            js += "    {\n      \"object_index\": " + py_json_str(r.index) +
                  ",\n      \"xmin\": " + py_float(r.box[0]) +
                  ",\n      \"ymin\": " + py_float(r.box[1]) +
                  ",\n      \"xmax\": " + py_float(r.box[2]) +
                  ",\n      \"ymax\": " + py_float(r.box[3]) +
                  ",\n      \"label\": " + py_json_str(r.label) + "\n    }" +
                  (i + 1 < out.boxes.size() ? ",\n" : "\n");
        }
        js += "  ]";
    }
    js += ",\n  \"prompt_tokens\": " + std::to_string(out.prompt_tokens) +
          ",\n  \"completion_tokens\": " +
          std::to_string(out.completion_tokens) + "\n}";
    out.json = js;

    // retain for the result getters
    s.answer = out.answer;
    s.result_json = out.json;
    s.boxes = out.boxes;
    s.prompt_tokens = out.prompt_tokens;
    s.completion_tokens = out.completion_tokens;
    s.proposal_count = out.proposal_count;
    s.preprocess_ms = out.preprocess_ms;
    s.vision_ms = out.vision_ms;
    s.llm_ms = out.llm_ms;
    s.postprocess_ms = out.postprocess_ms;
    s.serialization_ms = out.serialization_ms;
    s.e2e_ms = out.e2e_ms;
    s.has_result = true;
    return out;
}

bool VlxSeekRuntime::render_png(const VlxRunResult& result,
                                const uint8_t* rgb,
                                int width,
                                int height,
                                const std::string& out_png_path) {
    if (!rgb || width <= 0 || height <= 0 || out_png_path.empty()) return false;
    std::vector<uint8_t> canvas(rgb, rgb + size_t(width) * height * 3);
    for (const auto& r : result.boxes) {
        draw_rect_pil(&canvas, width, height, r.box[0], r.box[1], r.box[2],
                      r.box[3], 3);
        const std::string& label = all_ascii(r.label) ? r.label : r.index;
        draw_label_pil(&canvas, width, height, r.box[0], r.box[1], label);
    }
    return stbi_write_png(out_png_path.c_str(), width, height, 3, canvas.data(),
                          width * 3) != 0;
}

std::string VlxSeekRuntime::info_json() const {
    const Impl& s = *impl_;
    std::string js = "{\n";
    js += "  \"device\": " + py_json_str(s.device_resolved) + ",\n";
    js += "  \"threads\": " + std::to_string(s.threads) + ",\n";
    js += "  \"precision\": " + py_json_str(s.opts.precision) + ",\n";
    js += "  \"ready\": " + std::string(s.ready ? "true" : "false") + ",\n";
    js += "  \"last_run\": {\n";
    js += "    \"has_result\": " + std::string(s.has_result ? "true" : "false");
    if (s.has_result) {
        js += ",\n    \"prompt_tokens\": " + std::to_string(s.prompt_tokens);
        js += ",\n    \"completion_tokens\": " +
              std::to_string(s.completion_tokens);
        js += ",\n    \"boxes\": " + std::to_string(s.boxes.size());
        js += ",\n    \"proposals\": " + std::to_string(s.proposal_count);
        js += ",\n    \"vision_ms\": " + py_float(s.vision_ms);
        js += ",\n    \"llm_ms\": " + py_float(s.llm_ms);
        js += ",\n    \"e2e_ms\": " + py_float(s.e2e_ms);
    }
    js += "\n  }\n}\n";
    return js;
}

}  // namespace vlx
