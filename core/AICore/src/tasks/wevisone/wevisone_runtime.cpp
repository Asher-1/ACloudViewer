// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/wevisone/wevisone_runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/ggml_backend_utils.hpp"
#include "common/llama_norm_parity.hpp"
#include "common/model_cache.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "tasks/wevisone/wevisone_image_prep.h"
#include "tasks/wevisone/wevisone_mask_decoder_graph.h"
#include "tasks/wevisone/wevisone_pipeline_common.h"
#include "tasks/wevisone/wevisone_result_parser.h"
#include "tasks/wevisone/wevisone_vision_taps_graph.h"

#define WWV_LOG_DEBUG(...) ::wwv::logf(wwv::LogLevel::kDebug, __VA_ARGS__)
#define WWV_LOG_INFO(...) ::wwv::logf(wwv::LogLevel::kInfo, __VA_ARGS__)
#define WWV_LOG_WARN(...) ::wwv::logf(wwv::LogLevel::kWarn, __VA_ARGS__)
#define WWV_LOG_ERROR(...) ::wwv::logf(wwv::LogLevel::kError, __VA_ARGS__)

namespace wwv {

namespace {

// llama/mtmd WARN notices split into two classes:
//  * Benign families, verified against the pristine upstream CLI on
//    identical inputs (same messages, same numbers, same images):
//    - vocab control-token override (llama-vocab.cpp): inherent to the
//      pinned GGUF metadata; the loader self-heals the token type.
//    - Qwen-VL ">=1024 image tokens" advice (clip.cpp): generic llama.cpp
//      guidance. Upstream pins the 16/2025 merged-token window
//      (engine.py min/max_pixels) and the greedy parity gate passes at
//      that window, so the advice does not apply to this task.
//    - ~llama_context compute-buffer mismatch: the init reserve models
//      only the causal graph; the non-causal vision span of large images
//      (image tokens > n_ubatch) legitimately reallocs the compute buffer
//      (sched_graph_realloc). The pristine upstream CLI emits the
//      identical WARN with identical MiB values on the same image, and
//      inference parity is unaffected.
//    These are demoted to DEBUG: silent at the default console level,
//    recoverable through the task log_level option.
//  * Any other WARN: surfaced once per process (digits stripped so the
//    same notice with different numbers collapses to one key). ERROR and
//    first occurrences always reach the shared bridge.
void wevisone_log_forward(ggml_log_level level, const char* text, void* user) {
    if (level == GGML_LOG_LEVEL_WARN && text != nullptr) {
        if (std::strstr(text, "was not control-type") != nullptr ||
            std::strstr(text, "require at minimum 1024 image tokens") !=
                    nullptr ||
            std::strstr(text, "try adding --image-min-tokens") != nullptr ||
            std::strstr(text, "llama.cpp/issues/16842") != nullptr ||
            std::strstr(text, "does not match expectation of") != nullptr) {
            ggml_common::log_bridge_callback()(GGML_LOG_LEVEL_DEBUG, text,
                                               user);
            return;
        }
        std::string key;
        for (const char* p = text; *p != '\0'; ++p) {
            if (!std::isdigit(static_cast<unsigned char>(*p))) key += *p;
        }
        static std::mutex mu;
        static std::unordered_set<std::string> seen;
        std::lock_guard<std::mutex> lock(mu);
        if (!seen.insert(std::move(key)).second) return;
    }
    ggml_common::log_bridge_callback()(level, text, user);
}

// Chunked prefill that declares EVERY position as an output, matching the
// embeddings=true context (per-position hidden states for the pixel
// decoder). mtmd_helper_eval_chunks marks only the last token of each
// batch, which makes llama warn and override per batch; the overridden
// graph is identical to this explicit declaration, so greedy parity is
// unchanged. Mirrors the helper's text chunking, M-RoPE image positions
// and non-causal vision span.
int32_t eval_chunks_all_outputs(mtmd_context* mctx,
                                llama_context* lctx,
                                const mtmd_input_chunks* chunks,
                                llama_pos n_past,
                                int32_t n_batch,
                                llama_pos* new_n_past) {
    const int n_embd_inp = llama_model_n_embd_inp(llama_get_model(lctx));
    const bool mrope = mtmd_decode_use_mrope(mctx);
    const int n_pos_per_embd = mrope ? 4 : 1;
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t c = 0; c < n_chunks; ++c) {
        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const llama_token* tokens =
                    mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            llama_batch_ext* batch = llama_batch_ext_init(lctx);
            size_t i = 0;
            while (i < n_tokens) {
                llama_batch_ext_clear(batch);
                int32_t n_added = 0;
                for (; i < n_tokens && n_added < n_batch; ++i, ++n_added) {
                    const int32_t idx =
                            llama_batch_ext_add_token(batch, 0, tokens[i]);
                    const llama_pos pos = n_past + n_added;
                    llama_batch_ext_set_pos(batch, idx, &pos);
                    llama_batch_ext_set_output_logits(batch, idx, true);
                }
                const int32_t rc =
                        llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch);
                if (rc != 0) {
                    llama_batch_ext_free(batch);
                    return rc;
                }
                n_past += n_added;
            }
            llama_batch_ext_free(batch);
        } else {
            int32_t rc = mtmd_encode_chunk(mctx, chunk);
            if (rc != 0) return rc;
            const float* embd = mtmd_get_output_embd(mctx);
            const int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            // Section-major positions, mirroring decode_embd_batch.
            std::vector<llama_pos> pos(static_cast<size_t>(n_tokens) *
                                       n_pos_per_embd);
            if (mrope) {
                std::vector<mtmd_decoder_pos> rel(
                        static_cast<size_t>(n_tokens));
                mtmd_helper_image_get_decoder_pos(
                        mtmd_input_chunk_get_tokens_image(chunk), n_past,
                        rel.data());
                for (int32_t i = 0; i < n_tokens; ++i) {
                    pos[i] = rel[i].t;
                    pos[n_tokens + i] = rel[i].y;
                    pos[2 * n_tokens + i] = rel[i].x;
                    pos[3 * n_tokens + i] = rel[i].z;
                }
            } else {
                for (int32_t i = 0; i < n_tokens; ++i) pos[i] = n_past + i;
            }
            // The vision span attends non-causally (mtmd-helper contract).
            const bool non_causal = mtmd_decode_use_non_causal(mctx, chunk);
            if (non_causal) llama_set_causal_attn(lctx, false);
            llama_batch_ext* batch = llama_batch_ext_init(lctx);
            int32_t offset = 0;
            while (offset < n_tokens) {
                const int32_t n_add = std::min(n_batch, n_tokens - offset);
                llama_batch_ext_clear(batch);
                for (int32_t i = offset; i < offset + n_add; ++i) {
                    const llama_embd e = {
                            embd + static_cast<size_t>(i) * n_embd_inp, 1,
                            static_cast<size_t>(n_embd_inp)};
                    const int32_t idx = llama_batch_ext_add_embd(batch, 0, e);
                    llama_pos p[4] = {0, 0, 0, 0};
                    for (int j = 0; j < n_pos_per_embd; ++j) {
                        p[j] = pos[static_cast<size_t>(j) * n_tokens + i];
                    }
                    llama_batch_ext_set_pos(batch, idx, p);
                    llama_batch_ext_set_output_logits(batch, idx, true);
                }
                rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch);
                if (rc != 0) {
                    llama_batch_ext_free(batch);
                    if (non_causal) llama_set_causal_attn(lctx, true);
                    return rc;
                }
                offset += n_add;
            }
            llama_batch_ext_free(batch);
            if (non_causal) llama_set_causal_attn(lctx, true);
            n_past += mtmd_input_chunk_get_n_pos(chunk);
        }
    }
    *new_n_past = n_past;
    return 0;
}

std::string join_labels(const std::string& query) {
    // engine prompts.py: newline-separated labels are joined by ". ".
    // py _labels() keeps each label's INNER whitespace (item.strip() drops
    // leading/trailing whitespace only) — stripping inside the label was
    // corrupting multi-word labels ("the white king" -> "thewhiteking").
    std::vector<std::string> parts;
    std::string cur;
    std::istringstream in(query);
    while (std::getline(in, cur)) {
        size_t b = 0, e = cur.size();
        while (b < e && std::isspace(static_cast<unsigned char>(cur[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(cur[e - 1])))
            --e;
        if (e > b) parts.push_back(cur.substr(b, e - b));
    }
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += ". ";
        out += parts[i];
    }
    return out;
}

// py str.strip(): drop leading/trailing whitespace only, keep the middle.
std::string strip_edges(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// prompts.py _text(): the query text used verbatim (strip only). count_box /
// count_point / ocr_grounding pass this through WITHOUT the trailing-period
// handling _phrase applies — the previous phrase_of()+substr pair here ate a
// user-supplied trailing '.' that the python prompt keeps.
std::string text_of(const std::string& query) { return strip_edges(query); }

std::string phrase_of(const std::string& text) {
    std::string t;
    for (const char c : text) {
        if (!std::isspace(static_cast<unsigned char>(c)) || c == ' ') t += c;
    }
    // trim
    while (!t.empty() && t.front() == ' ') t.erase(t.begin());
    while (!t.empty() && t.back() == ' ') t.pop_back();
    if (!t.empty()) {
        const char last = t.back();
        if (last != '.' && last != '?' && last != '!') t += ".";
    }
    return t;
}

}  // namespace

std::string WevisoneRuntime::build_prompt(const std::string& task,
                                          const std::string& query,
                                          const float* ref_boxes,
                                          int32_t n_ref_boxes,
                                          int32_t width,
                                          int32_t height) {
    const std::string t = [](std::string s) {
        for (auto& c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }(task);
    const std::string labels = join_labels(query);
    if (t == "detection") {
        return "Locate all the instances that match the following "
               "description: " +
               labels + ".";
    }
    if (t == "referring") {
        return "Locate the instances that match the following description: " +
               phrase_of(query);
    }
    if (t == "point") return "point to: " + labels;
    if (t == "count_box") {
        const std::string q = text_of(query);
        return "How many " + (q.empty() ? "objects" : q) +
               " in the image? Please count them with boxes.";
    }
    if (t == "count_point") {
        const std::string q = text_of(query);
        return "How many " + (q.empty() ? "objects" : q) + " in the image?";
    }
    if (t == "segmentation" || t == "matting") {
        return "Locate the instances that match the following description and "
               "provide the alpha mattes: " +
               phrase_of(query);
    }
    if (t == "ocr") return "detect all the word in the box format";
    if (t == "ocr_grounding") {
        // prompts.py: "Locate the following text in the image: " + _text(query)
        // (stripped) — the raw query passed through here verbatim before.
        return "Locate the following text in the image: " + text_of(query);
    }
    if (t == "visual_prompt") {
        std::string coords;
        for (int32_t i = 0; i < n_ref_boxes; ++i) {
            // prompts.py _box_tokens skips a box when width/height <= 0 —
            // it never encodes garbage bins for a degenerate canvas.
            if (width <= 0 || height <= 0) break;
            const float* b = ref_boxes + 4 * i;
            const int v[4] = {
                    std::max(0,
                             std::min(1000, static_cast<int>(
                                                    b[0] / std::max(1, width) *
                                                    1000))),
                    std::max(0,
                             std::min(1000, static_cast<int>(
                                                    b[1] / std::max(1, height) *
                                                    1000))),
                    std::max(0,
                             std::min(1000, static_cast<int>(
                                                    b[2] / std::max(1, width) *
                                                    1000))),
                    std::max(0,
                             std::min(1000, static_cast<int>(
                                                    b[3] / std::max(1, height) *
                                                    1000)))};
            for (const int x : v) coords += "<" + std::to_string(x) + ">";
            if (i + 1 < n_ref_boxes) coords += ",";
        }
        return "Locate all similar objects in the image based on the reference "
               "boxes: <ref>object 1</ref><box>" +
               coords + "</box>";
    }
    return query;  // freeform
}

std::unique_ptr<WevisoneRuntime> WevisoneRuntime::create(
        const RuntimeOptions& opts, std::string* err) {
    std::unique_ptr<WevisoneRuntime> rt(new WevisoneRuntime());
    rt->opts_ = opts;
    wwv::set_log_level(static_cast<LogLevel>(opts.log_level));
    set_mkl_log_lib_path(opts.mkl_log_lib);
    if (!rt->load_models(err)) return nullptr;
    rt->ready_ = true;
    return rt;
}

bool WevisoneRuntime::resolve_device(std::string* err) {
    ggml_common::load_backends_once();
    const std::string want =
            ggml_common::to_lower(opts_.device.empty() ? "auto" : opts_.device);
    const bool force_cpu = want == "cpu";
    if (!force_cpu) {
        ggml_common::GpuBackendGroup group =
                ggml_common::resolve_gpu_group(want);
        if (group.primary()) {
            gpu_lease_ = aicore::runtime::adopt_backend_lease(
                    group.gpus[0], group.names[0], threads_);
            group.gpus[0] = nullptr;
            group.release();
        } else if (want != "auto") {
            if (err != nullptr)
                *err = "no ggml device for device '" + opts_.device + "'";
            return false;
        }
    }
    if (gpu_lease_) {
        cpu_mode_ = false;
        device_ = gpu_lease_.device();
    } else {
        std::string cpu_error;
        cpu_lease_ = aicore::runtime::acquire_backend_lease("cpu", threads_,
                                                            &cpu_error);
        if (!cpu_lease_) {
            if (err != nullptr)
                *err = cpu_error.empty() ? "CPU backend unavailable"
                                         : cpu_error;
            return false;
        }
        cpu_mode_ = true;
        device_ = cpu_lease_.device();
        if (want != "auto" && !force_cpu)
            WWV_LOG_WARN("requested device %s not found; using CPU",
                         opts_.device.c_str());
    }
    return true;
}

bool WevisoneRuntime::load_models(std::string* err) {
    threads_ =
            opts_.threads > 0
                    ? opts_.threads
                    : static_cast<int32_t>(ggml_common::default_cpu_threads());
    if (!resolve_device(err)) return false;
    if (!opts_.stages_dir.empty()) {
        // decoder-parity harness: no VLM, no mtmd — decoder weights only
        return load_decoder_weights(err);
    }
    precision_ = opts_.precision;
    const std::string models_dir = opts_.models_dir.empty()
                                           ? aicore::wevisone_model_cache_dir()
                                           : opts_.models_dir;

    std::string vlm_path = opts_.vlm_gguf;
    std::string dec_path = opts_.decoder_gguf;
    std::string mmproj_path = opts_.mmproj_gguf;
    if (vlm_path.empty()) {
        vlm_path = models_dir + "/wevisone-vlm-" + opts_.size + "-" +
                   opts_.precision + ".gguf";
    }
    if (mmproj_path.empty()) {
        // Only the f16 mmproj is published for each size (HF inventory);
        // the VLM quantization tier never changes the projector file.
        mmproj_path =
                models_dir + "/mmproj-wevisone-vlm-" + opts_.size + "-f16.gguf";
    }
    if (dec_path.empty()) {
        dec_path = models_dir + "/wevisone-mask-decoder-" + opts_.size + "-" +
                   opts_.decoder_precision + ".gguf";
    }

    // ---- GPU admission (-3 resource contract) ----
    auto file_bytes = [](const std::string& p) -> size_t {
        std::error_code ec;
        const auto sz =
                std::filesystem::file_size(std::filesystem::u8path(p), ec);
        return ec ? 0 : size_t(sz);
    };
    const size_t need = file_bytes(vlm_path) + file_bytes(mmproj_path) +
                        file_bytes(dec_path);
    std::string admission_error;
    if (need > 0 && !cpu_mode_ &&  // admission only applies to accelerators
        !ggml_common::gpu_admission_check(opts_.device, need,
                                          &admission_error)) {
        if (err != nullptr) *err = admission_error;
        return false;
    }

    // ---- llama/ggml logging follows the shared bridge ----
    // The task-local forwarder dedups model-inherent WARN notices to one
    // occurrence per process before handing off to the shared bridge.
    llama_log_set(wevisone_log_forward, nullptr);
    mtmd_helper_log_set(wevisone_log_forward, nullptr);

    llama_model_params mp = llama_model_default_params();
    // devices = {nullptr} means "no accelerator" (CPU-only model AND context
    // buffers); devices = nullptr would mean "no restriction" and let llama
    // place op buffers on an unrelated GPU even for CPU requests.
    ggml_backend_dev_t devices[2] = {
            cpu_mode_ ? nullptr : ggml_backend_get_device(gpu_lease_.handle()),
            nullptr};
    mp.devices = devices;
    mp.n_gpu_layers = cpu_mode_ ? 0 : -1;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_ = llama_model_load_from_file(vlm_path.c_str(), mp);
    if (model_ == nullptr) {
        if (err != nullptr) *err = "cannot load VLM gguf: " + vlm_path;
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_threads = threads_;
    cp.n_threads_batch = threads_;
    cp.no_perf = false;
    // per-token last-hidden-state readout for <OBJ> positions (t_seg):
    // embeddings=true outputs every position. pooling_type stays
    // UNSPECIFIED: llama resolves UNSPECIFIED + model-default(-1) to
    // POOLING_TYPE_NONE (llama-context.cpp), the exact semantics needed,
    // without the "model default pooling_type" WARN an explicit NONE trips.
    cp.embeddings = true;
    // WeVisOne's upstream pins the OLD scalar ggml RMS-norm kernel; tag the LLM
    // norm nodes so CUDA patch 0033 keeps the pinned accumulation order (greedy
    // parity). The ViT norms are unnamed and stay on the vectorized path.
    cp.cb_eval = ggml_common::llama_norm_parity_eval_cb;
    cp.cb_eval_user_data = nullptr;
    ctx_ = llama_init_from_model(model_, cp);
    if (ctx_ == nullptr) {
        if (err != nullptr) *err = "cannot create llama context";
        return false;
    }
    mtmd_context_params mtp = mtmd_context_params_default();
    mtp.n_threads = threads_;
    mtp.print_timings = false;
    mtp.warmup = false;
    mtp.use_gpu = !cpu_mode_;
    // Bind the ViT to the same lease-resolved device as the LLM. AICore
    // registers every compiled backend (CUDA + Vulkan) in one process, and
    // mtmd's default GPU pick (ggml_backend_init_by_type) returns the FIRST
    // registered GPU — CUDA0 even for a Vulkan request — which silently ran
    // the vision encoder on a different backend than the upstream
    // single-backend reference binary and broke greedy parity.
    mtp.device =
            cpu_mode_ ? nullptr : ggml_backend_get_device(gpu_lease_.handle());
    // Mirror engine.py's processor limits (16*32*32 / 2025*32*32 pixels) as
    // merged-token counts ((patch*merge)^2 = 1024 px per merged token).
    // mtmd's qwen3vl defaults (8/4096) resize bitmaps onto a DIFFERENT grid
    // than the mask decoder consumes — and than the official pytorch stack
    // does — so the vision-taps path would feed the decoder a foreign grid.
    mtp.image_min_tokens = 16;
    mtp.image_max_tokens = 2025;
    mtmd_ctx_ = mtmd_init_from_file(mmproj_path.c_str(), model_, mtp);
    if (mtmd_ctx_ == nullptr) {
        if (err != nullptr) *err = "mtmd init failed (mmproj missing in gguf?)";
        return false;
    }
    WWV_LOG_INFO("vlm=%s device=%s threads=%d", vlm_path.c_str(),
                 device_.c_str(), threads_);
    return true;
}

WevisoneRuntime::~WevisoneRuntime() {
    if (mtmd_ctx_ != nullptr) mtmd_free(mtmd_ctx_);
    if (ctx_ != nullptr) llama_free(ctx_);
    if (model_ != nullptr) llama_model_free(model_);
    // Session-local decoder graphs/buffers are freed per run; the physical
    // backend belongs to the shared lease registry.
    pixel_backend_ = nullptr;
    pixel_dev_ = nullptr;
    gpu_lease_.reset();
    cpu_lease_.reset();
}

bool WevisoneRuntime::warmup_backend() {
    if (ctx_ == nullptr) return false;
    return ctx_ != nullptr;
}

bool WevisoneRuntime::run_vlm(const RgbImage& rgb,
                              const std::string& prompt,
                              VlmOutput* out,
                              std::string* err) {
    if (ctx_ == nullptr || mtmd_ctx_ == nullptr) {
        if (err != nullptr) *err = "runtime not loaded";
        return false;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    // Released Qwen3-VL chat template with the image placeholder at the
    // mtmd media marker position.
    std::string chat =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><__media__><|vision_end|>" +
            prompt + "<|im_end|>\n<|im_start|>assistant\n";

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_bitmap* bmp = mtmd_bitmap_init(static_cast<uint32_t>(rgb.width),
                                        static_cast<uint32_t>(rgb.height),
                                        rgb.pixels.data());
    mtmd_input_text text{chat.c_str(), chat.size(), false, true};
    const mtmd_bitmap* bmps[1] = {bmp};
    int32_t rc = mtmd_tokenize(mtmd_ctx_, chunks, &text, bmps, 1);
    mtmd_bitmap_free(bmp);
    if (rc != 0) {
        if (err != nullptr) *err = "mtmd tokenize failed";
        mtmd_input_chunks_free(chunks);
        return false;
    }
    llama_memory_clear(llama_get_memory(ctx_), false);
    llama_pos n_past = 0;
    rc = eval_chunks_all_outputs(mtmd_ctx_, ctx_, chunks, 0,
                                 static_cast<int32_t>(512), &n_past);
    mtmd_input_chunks_free(chunks);
    if (rc != 0) {
        if (err != nullptr) *err = "mtmd eval (chat) failed";
        return false;
    }
    out->n_chat = static_cast<int32_t>(n_past);

    // greedy decode over raw logits — explicit parity contract
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_token obj_id = -1;
    llama_tokenize(vocab, "<OBJ>", 5, &obj_id, 1, false, true);
    while (static_cast<int32_t>(out->ids.size()) < opts_.max_new_tokens) {
        const float* logits = llama_get_logits_ith(ctx_, -1);
        int32_t best = 0;
        for (int32_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        if (llama_vocab_is_eog(vocab, best)) break;
        char buf[64];
        const int n =
                llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
        if (n > 0) out->text.append(buf, static_cast<size_t>(n));
        out->ids.push_back(best);
        llama_token one = best;
        llama_batch next = llama_batch_get_one(&one, 1);
        if (llama_decode(ctx_, next) != 0) {
            if (err != nullptr) *err = "decode step failed";
            return false;
        }
        ++n_past;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Pixel-decoder pass state: weight files + backend (loaded lazily so a pure
// object-level deployment never reads the decoder GGUF).
// ---------------------------------------------------------------------------

namespace {

// Host sine 2-D position embedding (DINO/DETR semantics, mask_decoder.py
// PositionEmbeddingSine2D): half channels for y, half for x, interleaved
// sin/cos; the level embedding is added in-graph. Layout (d, levels*hw),
// level blocks in the same order as the token concat. O(levels*d*hw).
std::vector<float> build_sine_pos_leaf(int32_t d,
                                       int32_t grid_w,
                                       int32_t grid_h,
                                       int32_t levels) {
    const int32_t half = d / 2;  // num_pos_feats
    const int32_t hw = grid_w * grid_h;
    std::vector<float> pos(static_cast<size_t>(levels) * d * hw, 0.0f);
    std::vector<double> dim_t(static_cast<size_t>(half));
    for (int32_t i = 0; i < half; ++i) {
        dim_t[static_cast<size_t>(i)] =
                std::pow(10000.0, 2.0 * (i / 2) / static_cast<double>(half));
    }
    // PositionEmbeddingSine2D: pos = cat([pos_y, pos_x], -1) — the first
    // half of the channels encodes y, the second half x; channel j is
    // sin(coord/dim_t[j]) for even j and cos(coord/dim_t[j]) for odd j.
    for (int32_t l = 0; l < levels; ++l) {
        float* base = pos.data() + static_cast<size_t>(l) * d * hw;
        for (int32_t y = 0; y < grid_h; ++y) {
            const double ny = (y + 0.5) / grid_h * 2.0 * M_PI;
            for (int32_t x = 0; x < grid_w; ++x) {
                const double nx = (x + 0.5) / grid_w * 2.0 * M_PI;
                for (int32_t i = 0; i < d; ++i) {
                    const double v = (i < half) ? ny : nx;
                    const float e = static_cast<float>(
                            v / dim_t[static_cast<size_t>(i % half)]);
                    base[static_cast<size_t>(i) * hw + y * grid_w + x] =
                            (i % 2 == 0) ? std::sin(e) : std::cos(e);
                }
            }
        }
    }
    return pos;
}

}  // namespace

bool WevisoneRuntime::load_decoder_weights(std::string* err) {
    if (decoder_gguf_ != nullptr) return true;
    const std::string models_dir = opts_.models_dir.empty()
                                           ? aicore::wevisone_model_cache_dir()
                                           : opts_.models_dir;
    std::string dec_path = opts_.decoder_gguf;
    if (dec_path.empty()) {
        dec_path = models_dir + "/wevisone-mask-decoder-" + opts_.size + "-" +
                   opts_.decoder_precision + ".gguf";
    }
    std::string mmproj_path = opts_.mmproj_gguf;
    if (mmproj_path.empty()) {
        mmproj_path =
                models_dir + "/mmproj-wevisone-vlm-" + opts_.size + "-f16.gguf";
    }
    decoder_gguf_ = std::make_unique<GgufFile>();
    if (!decoder_gguf_->load(dec_path, err)) {
        decoder_gguf_.reset();
        return false;
    }
    // Physical backend comes from the lease resolved in load_models();
    // graphs and buffers stay session-local (per-run gallocr / host pool).
    pixel_backend_ = cpu_mode_ ? cpu_lease_.handle() : gpu_lease_.handle();
    pixel_dev_ = ggml_backend_get_device(pixel_backend_);
    if (pixel_backend_ == nullptr || pixel_dev_ == nullptr) {
        if (err != nullptr) *err = "no decode backend for " + device_;
        return false;
    }
    if (!opts_.stages_dir.empty()) {
        // stages-in harness: the levels arrive precomputed, no ViT needed.
        return true;
    }
    // ViT weights live in the mmproj GGUF; keep only the "v." subtree so
    // unrelated mmproj metadata never enters the decode graph.
    vision_gguf_ = std::make_unique<GgufFile>();
    const std::vector<std::string> prefixes = {"v."};
    if (!vision_gguf_->load(mmproj_path, err, &prefixes)) {
        vision_gguf_.reset();
        return false;
    }
    WWV_LOG_INFO("pixel backend: %s", ggml_backend_name(pixel_backend_));
    return true;
}

bool WevisoneRuntime::run_image(const aicore_image_view& img,
                                const std::string& task,
                                const std::string& query,
                                const float* ref_boxes,
                                int32_t n_ref_boxes,
                                RunResult* out,
                                const std::string& prompt_override) {
    if (!opts_.stages_dir.empty()) {
        // Decoder-parity harness: the boxes come from the torch dump.
        static const std::vector<ResultBox> kBoxes = {};
        std::string serr;
        const bool ok = run_stages_in(opts_.stages_dir, kBoxes, out, &serr);
        if (!ok && !serr.empty()) last_error_ = serr;
        return ok;
    }
    const double t0 = now_ms();
    RgbImage rgb;
    std::string err;
    if (!view_to_rgb(img, &rgb, &err)) {
        last_error_ = err;
        return false;
    }
    const std::string prompt =
            prompt_override.empty()
                    ? build_prompt(task, query, ref_boxes, n_ref_boxes,
                                   rgb.width, rgb.height)
                    : prompt_override;
    const double t1 = now_ms();
    VlmOutput vout;
    if (!run_vlm(rgb, prompt, &vout, &err)) {
        last_error_ = err;
        return false;
    }
    const std::string answer = vout.text;
    // <OBJ> positions among the generated ids
    std::vector<int32_t> obj_positions;
    {
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        llama_token obj_id = -1;
        if (llama_tokenize(vocab, "<OBJ>", 5, &obj_id, 1, false, true) == 1) {
            for (size_t i = 0; i < vout.ids.size(); ++i) {
                if (vout.ids[i] == obj_id) {
                    obj_positions.push_back(static_cast<int32_t>(i));
                }
            }
        }
    }
    const double t2 = now_ms();
    const ParsedAnswer parsed = parse_answer(answer, rgb.width, rgb.height);
    out->answer = parsed.answer;
    out->boxes = parsed.boxes_px;
    out->has_obj = parsed.has_obj;
    out->prompt_tokens = vout.n_chat;
    out->completion_tokens = static_cast<int32_t>(vout.ids.size());
    const double t3 = now_ms();
    out->timings = make_timings(t1 - t0, t2 - t1, t3 - t2, 0.0, t3 - t0);
    if (!opts_.decode_masks || obj_positions.empty()) {
        return true;
    }
    std::string perr;
    if (!run_pixel_pass(rgb, prompt, answer, obj_positions, out->boxes, out,
                        &perr)) {
        last_error_ = perr;
        return false;
    }
    const double t4 = now_ms();
    // timing contract (aicore_wevisone_pipeline_timing.h): the teacher-forced
    // hidden-state pass and the pixel-decoder execution/readback are part
    // of inference — without folding the pixel pass in, the matting path
    // reported a ~2.3 s unaccounted gap between pre+infer+post and e2e.
    out->timings =
            make_timings(t1 - t0, t2 - t1 + (t4 - t3), t3 - t2, 0.0, t4 - t0);
    return true;
}

// engine.decode_masks: teacher-force prompt + response (no generation) and
// decode mattes from the <OBJ> hidden states. boxes01 (normalized xyxy)
// overrides the parsed boxes — the GT-conditioned evaluation protocol.
bool WevisoneRuntime::run_decode_masks(const aicore_image_view& img,
                                       const std::string& prompt,
                                       const std::string& response,
                                       const float* boxes01,
                                       int32_t n_boxes,
                                       RunResult* out) {
    const double t0 = now_ms();
    RgbImage rgb;
    std::string err;
    if (!view_to_rgb(img, &rgb, &err)) {
        last_error_ = err;
        return false;
    }
    if (n_boxes < 0 || (n_boxes > 0 && boxes01 == nullptr)) {
        last_error_ = "boxes01 is required when n_boxes > 0";
        return false;
    }
    // <OBJ> count in the teacher-forced sequence: the decode query budget.
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    llama_token obj_id = -1;
    if (llama_tokenize(vocab, "<OBJ>", 5, &obj_id, 1, false, true) != 1) {
        last_error_ = "<OBJ> not in vocab";
        return false;
    }
    std::vector<llama_token> resp_toks(response.size() + 8);
    const int n_resp = llama_tokenize(
            vocab, response.c_str(), static_cast<int32_t>(response.size()),
            resp_toks.data(), static_cast<int32_t>(resp_toks.size()), false,
            true);
    if (n_resp <= 0) {
        last_error_ = "response tokenize failed";
        return false;
    }
    int32_t n_obj = 0;
    for (int i = 0; i < n_resp; ++i) {
        if (resp_toks[static_cast<size_t>(i)] == obj_id) ++n_obj;
    }

    // Conditioning boxes: the explicit override (unrounded GT protocol) or
    // the response tokens (rounded, the generated-path semantics).
    std::vector<ResultBox> cond_boxes;
    if (boxes01 != nullptr && n_boxes > 0) {
        if (n_obj <= 0) {
            // python decode_masks raises RuntimeError for exactly this case.
            last_error_ = "the teacher-forced response contains no <OBJ> token";
            return false;
        }
        const int32_t count = std::min(n_boxes, n_obj);
        cond_boxes.reserve(static_cast<size_t>(count));
        for (int32_t k = 0; k < count; ++k) {
            const float* b = boxes01 + 4 * k;
            ResultBox rb;
            // The px boxes below are only for reporting (the C API contract
            // exposes boxes in original-image pixels); the decoder consumes
            // the exact unrounded floats via boxes01_direct_, bypassing the
            // px round-trip and the 0-1 clamp of the parsed path — python
            // passes the GT box raw.
            rb.x1 = b[0] * rgb.width;
            rb.y1 = b[1] * rgb.height;
            rb.x2 = b[2] * rgb.width;
            rb.y2 = b[3] * rgb.height;
            cond_boxes.push_back(rb);
        }
        boxes01_direct_.assign(boxes01, boxes01 + 4 * count);
        cond_boxes.resize(static_cast<size_t>(count));
    } else {
        // boxes=None: engine.decode_masks parses them from the response and
        // returns [], [] when nothing survives (keep only the <OBJ>-preceded
        // groups — parse_seg_boxes semantics; parse_answer marks has_obj).
        const ParsedAnswer parsed =
                parse_answer(response, rgb.width, rgb.height);
        for (size_t k = 0; k < parsed.boxes_px.size(); ++k) {
            if (parsed.has_obj[k]) cond_boxes.push_back(parsed.boxes_px[k]);
        }
        if (cond_boxes.empty()) {
            out->answer = response;
            out->masks_decoded = false;
            out->prompt_tokens = 0;
            out->completion_tokens = n_resp;
            const double t_done = now_ms();
            out->timings =
                    make_timings(t_done - t0, 0.0, 0.0, 0.0, t_done - t0);
            return true;
        }
        if (n_obj <= 0) {
            last_error_ = "the teacher-forced response contains no <OBJ> token";
            return false;
        }
    }

    // count = min(len(boxes), n_seg) — the python truncation contract; both
    // the boxes and the t_seg rows are cut to it inside the pixel pass.
    const int32_t count =
            std::min<int32_t>(static_cast<int32_t>(cond_boxes.size()), n_obj);
    out->boxes = std::move(cond_boxes);
    out->boxes.resize(static_cast<size_t>(count));
    out->has_obj.assign(static_cast<size_t>(count), true);
    out->answer = response;
    out->prompt_tokens = 0;
    out->completion_tokens = n_resp;

    const double t1 = now_ms();
    // obj_positions only feeds run_pixel_pass's query count (the response
    // token scan inside re-derives the true embedding positions).
    std::vector<int32_t> q_provider(static_cast<size_t>(count), 0);
    if (!run_pixel_pass(rgb, prompt, response, q_provider, out->boxes, out,
                        &err)) {
        last_error_ = err;
        return false;
    }
    const double t2 = now_ms();
    // Timing contract (aicore_wevisone_pipeline_timing.h): the teacher-forced
    // pass and the decoder execution are inference time; no generation
    // happened.
    out->timings = make_timings(t1 - t0, t2 - t1, 0.0, 0.0, t2 - t0);
    return true;
}

bool WevisoneRuntime::run_stages_in(const std::string& dir,
                                    const std::vector<ResultBox>& boxes,
                                    RunResult* out,
                                    std::string* err) {
    // The torch dump carries the normalized xyxy boxes the reference run
    // used; feed them straight into the decoder (bypasses pixel-box
    // normalization, which makes no sense for a harness).
    {
        std::FILE* f = std::fopen((dir + "/boxes01.torch.bin").c_str(), "rb");
        if (f != nullptr) {
            boxes01_direct_.clear();
            float v;
            while (std::fread(&v, 4, 1, f) == 1) boxes01_direct_.push_back(v);
            std::fclose(f);
        }
    }
    // optional MSDA isolation override: torch's own query position
    // (MaskQueryBuilder output). When present it replaces the in-graph
    // box-Fourier branch so the cross-attention parity isolates the MSDA
    // math from the query_position band.
    query_position_direct_.clear();
    {
        // both dump conventions: the gate dir stores "tap_<stage>" names
        std::FILE* f =
                std::fopen((dir + "/query_position.torch.bin").c_str(), "rb");
        if (f == nullptr) {
            f = std::fopen((dir + "/tap_query_position.torch.bin").c_str(),
                           "rb");
        }
        if (f != nullptr) {
            float v;
            while (std::fread(&v, 4, 1, f) == 1)
                query_position_direct_.push_back(v);
            std::fclose(f);
        }
    }
    auto load_stage = [&](const std::string& name)
            -> std::pair<std::vector<float>, std::vector<int64_t>> {
        std::FILE* f = std::fopen((dir + "/" + name + ".shape").c_str(), "r");
        if (f == nullptr) {
            if (err != nullptr) *err = "missing stage " + name;
            return {};
        }
        char buf[256] = {};
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        std::vector<int64_t> shape;
        size_t k = 0;
        while (k < n) {
            char* end = nullptr;
            long v = std::strtol(buf + k, &end, 10);
            if (end == buf + k) break;
            shape.push_back(v);
            k = static_cast<size_t>(end - buf);
            if (k < n && buf[k] == ',') ++k;
        }
        f = std::fopen((dir + "/" + name + ".torch.bin").c_str(), "rb");
        if (f == nullptr) {
            if (err != nullptr) *err = "missing stage " + name;
            return {};
        }
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<float> host(static_cast<size_t>(sz) / 4);
        if (std::fread(host.data(), 4, host.size(), f) != host.size()) {
            std::fclose(f);
            if (err != nullptr) *err = "short read " + name;
            return {};
        }
        std::fclose(f);
        return {std::move(host), shape};
    };
    auto [t_seg_h, t_seg_shape] = load_stage("t_seg");
    auto [cnn_h, cnn_shape] = load_stage("cnn_image");
    auto [lv0, lv_shape] = load_stage("level0");
    if (t_seg_shape.empty() || cnn_shape.empty() || lv_shape.empty()) {
        return false;
    }
    std::vector<std::vector<float>> level_maps;
    level_maps.push_back(std::move(lv0));
    for (int l = 1; l < cfg_.msda_num_levels; ++l) {
        level_maps.push_back(load_stage("level" + std::to_string(l)).first);
    }
    const int32_t q = static_cast<int32_t>(t_seg_shape[0]);
    const int64_t n_embd = t_seg_shape[1];
    const int64_t C = lv_shape[0];
    const int64_t map_h = lv_shape[1];
    const int64_t map_w = lv_shape[2];
    // S follows the dumped cnn canvas (the parity gate may run at a reduced
    // cnn_image_size; the decoder semantics are size-invariant).
    const int64_t S = cnn_shape[1];
    if (rgb_for_stages_.width == 0) {
        rgb_for_stages_.width = static_cast<int32_t>(S);
        rgb_for_stages_.height = static_cast<int32_t>(S);
        rgb_for_stages_.pixels.assign(static_cast<size_t>(S) * S * 3, 0.0f);
    }
    return run_decoder_graph(rgb_for_stages_, std::move(cnn_h),
                             std::move(level_maps), std::move(t_seg_h), boxes,
                             q, S, map_w, map_h, C, n_embd, out, err);
}

bool WevisoneRuntime::run_pixel_pass(const RgbImage& rgb,
                                     const std::string& prompt,
                                     const std::string& response,
                                     const std::vector<int32_t>& obj_positions,
                                     const std::vector<ResultBox>& boxes,
                                     RunResult* out,
                                     std::string* err) {
    if (!load_decoder_weights(err)) return false;
    const int32_t q = static_cast<int32_t>(obj_positions.size());
    if (q == 0 || boxes.empty()) {
        out->masks_decoded = false;
        return true;  // rejection path: nothing to decode
    }

    // ---- 1. preprocessing -------------------------------------------------
    const double perf_t0 = now_ms();
    const int32_t patch = 16;
    const int32_t merge = 2;
    // engine.py processor limits (16*32*32 / 2025*32*32 pixels). The mtmd
    // tower must resize the raw bitmap onto the SAME patch grid the decoder
    // consumes, so load_models() mirrors these limits as merged-token
    // counts ((patch*merge)^2 = 1024 px per merged token) — the dims here
    // are the shared contract, asserted against the taps in step 3.
    const int64_t kMinPixels = static_cast<int64_t>(16) * 32 * 32;
    const int64_t kMaxPixels = static_cast<int64_t>(2025) * 32 * 32;
    int32_t vit_w = 0;
    int32_t vit_h = 0;
    smart_resize_dims(rgb.width, rgb.height, patch * merge, kMinPixels,
                      kMaxPixels, &vit_w, &vit_h);
    const int64_t npx = vit_w / patch;  // full patch-grid width
    const int64_t npy = vit_h / patch;
    const int64_t n_tok = npx * npy;
    // The bisect dump path still runs the torch-exact manual ViT and needs
    // its normalized pixels; the production path never materializes them.
    VitInput vit;
    if (!opts_.dump_dir.empty()) {
        vit = smart_resize_normalized(rgb, patch, merge, kMinPixels,
                                      kMaxPixels);
        if (vit.width != vit_w || vit.height != vit_h) {
            if (err != nullptr) *err = "vit grid mismatch (dump path)";
            return false;
        }
    }
    std::vector<float> cnn = bicubic_square_chw(rgb, cfg_.cnn_image_size);
    const int64_t S = cfg_.cnn_image_size;
    const int64_t s2 = S / 2;
    const int64_t s1 = S;

    // Arm the mtmd vision taps for the production path: the teacher-forced
    // eval below encodes the image on the mtmd tower, and the armed taps
    // capture the decoder feature layers in the same forward (round-13:
    // the manual ViT replay is retired; the guard disarms on every exit so
    // a later generation-pass encode never pays the D2H readback).
    const bool taps_via_mtmd = opts_.dump_dir.empty();
    struct TapsGuard {
        mtmd_context* ctx;
        bool active = false;
        ~TapsGuard() {
            if (active) mtmd_vision_taps_configure(ctx, nullptr, 0);
        }
    } taps_guard{mtmd_ctx_, false};
    if (taps_via_mtmd) {
        if (mtmd_vision_taps_configure(
                    mtmd_ctx_, cfg_.vision_feature_layers.data(),
                    static_cast<size_t>(cfg_.vision_feature_layers.size())) !=
            0) {
            if (err != nullptr) *err = "vision taps configure failed";
            return false;
        }
        taps_guard.active = true;
    }

    // ---- 2. teacher-forced t_seg -----------------------------------------
    const double perf_t1 = now_ms();  // preprocess done
    // Re-eval the chat chunks (image seen exactly as in generation) and then
    // decode the response tokens in one batch; POOLING_TYPE_NONE +
    // embeddings=true exposes the last hidden state of every position, so
    // the <OBJ> rows are t_seg (engine.py LastHiddenCapture).
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    llama_token obj_id = -1;
    if (llama_tokenize(vocab, "<OBJ>", 5, &obj_id, 1, false, true) != 1) {
        if (err != nullptr) *err = "<OBJ> not in vocab";
        return false;
    }
    std::vector<llama_token> resp_toks(response.size() + 8);
    const int n_resp = llama_tokenize(
            vocab, response.c_str(), static_cast<int32_t>(response.size()),
            resp_toks.data(), static_cast<int32_t>(resp_toks.size()), false,
            true);
    if (n_resp <= 0) {
        if (err != nullptr) *err = "response tokenize failed";
        return false;
    }
    resp_toks.resize(static_cast<size_t>(n_resp));
    std::vector<int32_t> obj_seq_pos;
    for (size_t i = 0; i < resp_toks.size(); ++i) {
        if (resp_toks[i] == obj_id) {
            obj_seq_pos.push_back(static_cast<int32_t>(i));
        }
    }
    if (obj_seq_pos.empty()) {
        out->masks_decoded = false;
        return true;
    }
    const int32_t n_embd = llama_model_n_embd(model_);
    std::vector<float> t_seg(static_cast<size_t>(obj_seq_pos.size()) * n_embd);
    {
        llama_memory_clear(llama_get_memory(ctx_), false);
        mtmd_input_chunks* chat_chunks = mtmd_input_chunks_init();
        std::string chat =
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><__media__><|vision_end|>" +
                prompt + "<|im_end|>\n<|im_start|>assistant\n";
        mtmd_bitmap* bmp = mtmd_bitmap_init(static_cast<uint32_t>(rgb.width),
                                            static_cast<uint32_t>(rgb.height),
                                            rgb.pixels.data());
        mtmd_input_text text{chat.c_str(), chat.size(), false, true};
        const mtmd_bitmap* bmps[1] = {bmp};
        int32_t rc = mtmd_tokenize(mtmd_ctx_, chat_chunks, &text, bmps, 1);
        mtmd_bitmap_free(bmp);
        llama_pos n_past = 0;
        if (rc == 0) {
            rc = eval_chunks_all_outputs(mtmd_ctx_, ctx_, chat_chunks, 0, 512,
                                         &n_past);
        }
        mtmd_input_chunks_free(chat_chunks);
        if (rc != 0) {
            if (err != nullptr) *err = "teacher-force chat eval failed";
            return false;
        }
        // Same auto-tracked positions as llama_batch_get_one (upstream
        // contract), but every response position declared as an output so
        // the <OBJ> hidden-state readback needs no llama override WARN.
        llama_batch batch = llama_batch_get_one(
                resp_toks.data(), static_cast<int32_t>(resp_toks.size()));
        std::vector<int8_t> all_outputs(static_cast<size_t>(n_resp), 1);
        batch.logits = all_outputs.data();
        if (llama_decode(ctx_, batch) != 0) {
            if (err != nullptr) *err = "teacher-force decode failed";
            return false;
        }
        for (size_t k = 0; k < obj_seq_pos.size(); ++k) {
            const float* e = llama_get_embeddings_ith(
                    ctx_, obj_seq_pos[static_cast<size_t>(k)]);
            if (e == nullptr) {
                if (err != nullptr) *err = "embedding readback failed";
                return false;
            }
            std::memcpy(t_seg.data() + k * n_embd, e,
                        sizeof(float) * static_cast<size_t>(n_embd));
        }
    }

    // ---- 3. decoder feature taps ------------------------------------------
    const double perf_t2 = now_ms();  // teacher force done (taps captured)
    const int32_t levels = cfg_.msda_num_levels;
    std::vector<std::vector<float>> level_maps(levels);
    int64_t n_embd_vit = 0;
    if (taps_via_mtmd) {
        // Production path: the teacher-forced eval already encoded the image
        // on the mtmd tower with taps armed — one ViT forward serves both
        // the LLM and the decoder (round-13 retires the manual replay; the
        // D2H tap readback happens inside the mtmd encode itself, so only
        // the host un-shuffle remains in this segment).
        std::vector<mtmd_vision_tap> taps(static_cast<size_t>(levels));
        const int32_t n_taps =
                mtmd_vision_taps_get(mtmd_ctx_, taps.data(), levels);
        if (n_taps != levels) {
            if (err != nullptr) *err = "mtmd vision taps missing after encode";
            return false;
        }
        n_embd_vit = taps[0].n_embd;
        const int64_t C = n_embd_vit;
        const int64_t map_w = npx / merge;  // block grid
        const int64_t map_h = npy / merge;
        for (int32_t li = 0; li < levels; ++li) {
            // Grid invariant: any tower/limit drift would silently move the
            // decoder onto a foreign token grid — fail loudly instead.
            if (taps[li].n_tokens != n_tok || taps[li].n_embd != C) {
                if (err != nullptr)
                    *err = "mtmd vit tap grid mismatch (t=" +
                           std::to_string(taps[li].n_tokens) + "/" +
                           std::to_string(n_tok) +
                           " c=" + std::to_string(taps[li].n_embd) + "/" +
                           std::to_string(C) + ")";
                return false;
            }
            level_maps[li].assign(static_cast<size_t>(C) * n_tok, 0.0f);
        }
        {
            // The four host un-shuffles are independent — run them on
            // parallel threads (the round-12 perf bisect showed the
            // serialized loop at ~310 ms; the tap data is already host-side
            // and read-only, safe to overlap).
            std::vector<std::thread> tap_workers;
            for (int32_t li = 0; li < levels; ++li) {
                auto work = [&, li] {
                    unshuffle_tokens_host(taps[li].data, C, n_tok, map_w, map_h,
                                          merge, level_maps[li].data());
                };
                if (li + 1 < levels) {
                    tap_workers.emplace_back(work);
                } else {
                    work();  // the last layer runs inline
                }
            }
            for (auto& w : tap_workers) w.join();
        }
        const double perf_t4 = now_ms();  // taps un-shuffled
        if (!run_decoder_graph(rgb, std::move(cnn), std::move(level_maps),
                               std::move(t_seg), boxes, q, S, npx, npy,
                               n_embd_vit, n_embd, out, err)) {
            return false;
        }
        // The tap D2H readback rides inside the mtmd encode (teacher
        // segment); vit_graph cost is gone entirely.
        WWV_LOG_INFO(
                "[wwv-perf] pixel pass: preprocess %.0f ms | teacher+taps %.0f "
                "ms | "
                "unshuffle %.0f ms | decoder %.0f ms",
                perf_t1 - perf_t0, perf_t2 - perf_t1, perf_t4 - perf_t2,
                now_ms() - perf_t4);
        return true;
    }

    // ---- 3b. bisect dump path: manual sentinel ViT (unchanged) ------------
    // ViT hyperparams from the mmproj metadata (fallbacks = released cfg).
    const int32_t n_blocks = [&] {
        int32_t n = 0;
        for (const auto& t : vision_gguf_->tensors()) {
            if (t.name.rfind("v.blk.", 0) == 0 &&
                t.name.find(".ln1.weight") != std::string::npos) {
                ++n;
            }
        }
        return n;
    }();
    const int32_t heads = static_cast<int32_t>(
            vision_gguf_->i32("clip.vision_attention_heads", 16));
    const float eps = static_cast<float>(
            vision_gguf_->f32v("clip.vision_attention_layernorm_eps", 1e-6));
    // grid dims come from the shared smart-resize contract (outer scope)
    const int32_t vit_channels = static_cast<int32_t>(
            vision_gguf_->i32("clip.vision_projection_dim", 0));
    // fall back to the patch-embed output channels
    const auto* pe0 = vision_gguf_->tensor("v.patch_embd.weight");
    n_embd_vit = vit_channels > 0 && pe0 == nullptr
                         ? vit_channels
                         : (pe0 != nullptr ? static_cast<int32_t>(pe0->ne[3])
                                           : vit_channels);

    ggml_context* vctx = nullptr;
    {
        // generous context: graph + weight leaves for one ViT forward
        const size_t estimate = 64ull << 20;
        ggml_init_params ip{estimate, nullptr, true};
        vctx = ggml_init(ip);
    }

    VisionTapsParams vp;
    vp.grid_w = vit.width;
    vp.grid_h = vit.height;
    vp.patch_size = patch;
    vp.merge_size = merge;
    vp.heads = heads;
    vp.eps = eps;

    // bisect probes (dump path only): sentinel taps -1 (patch-embed+bias
    // output) and -2 (position-embed output) plus the first block output
    // (0) — raw (C, T) flats for the torch-side stage-by-stage diff. The
    // decoder keeps consuming the configured layers (cfg_tap0 below).
    std::vector<int32_t> tap_layers = cfg_.vision_feature_layers;
    if (!opts_.dump_dir.empty()) {
        tap_layers.insert(tap_layers.begin(),
                          {0, -2, -1, -3, -4, -5, -6, -7, -8});
    }
    std::vector<ggml_tensor*> tap_leaves;
    for (size_t i = 0; i < tap_layers.size(); ++i) {
        // the qkv-projection probe carries 3*C channels per token
        const int64_t ne0 = tap_layers[i] == -6 ? 3 * n_embd_vit : n_embd_vit;
        tap_leaves.push_back(
                ggml_new_tensor_2d(vctx, GGML_TYPE_F32, ne0, n_tok));
    }
    ggml_tensor* pixels =
            ggml_new_tensor_3d(vctx, GGML_TYPE_F32, vit.width, vit.height, 3);
    std::vector<int32_t> positions =
            vision_mrope_positions(vit.width, vit.height, patch, merge);
    ggml_tensor* pos_leaf = ggml_new_tensor_1d(
            vctx, GGML_TYPE_I32, static_cast<int64_t>(positions.size()));
    // no_alloc ctx: the bytes are uploaded after ggml_gallocr_alloc_graph
    // (staged-upload contract) — never write t->data here.

    std::vector<std::pair<ggml_tensor*, const void*>> staged_v;
    // Matmul/conv weights ride F16 straight from the GGUF: mul_mat and the
    // conv path dequantize on the fly — the exact numeric path mtmd's
    // clip.cpp runs its mmproj with — halving both the H2D upload and the
    // compute bandwidth vs the F32 expansion (the round-12 perf bisect
    // pinned the manual ViT at 784 ms under all-F32 weights). Norm/bias
    // parameters stay F32: the graph's binary ops are F32×F32.
    auto is_matmul_weight = [](const std::string& n) {
        return n.find("patch_embd.weight") != std::string::npos ||
               n.find("attn_qkv.weight") != std::string::npos ||
               n.find("attn_out.weight") != std::string::npos ||
               n.find("mlp.fc1.weight") != std::string::npos ||
               n.find("mlp.fc2.weight") != std::string::npos;
    };
    auto vw = [&](const char* name) -> ggml_tensor* {
        const auto* ti = vision_gguf_->tensor(name);
        if (ti == nullptr) return nullptr;
        if (ti->dtype == GgmlDType::F16 && is_matmul_weight(name)) {
            ggml_tensor* t =
                    ggml_new_tensor(vctx, GGML_TYPE_F16, ti->n_dims,
                                    reinterpret_cast<const int64_t*>(ti->ne));
            ggml_set_name(t, name);
            staged_v.emplace_back(t, ti->bytes.data());
            return t;
        }
        const float* f = vision_gguf_->tensor_f32(name);
        if (f == nullptr) return nullptr;
        ggml_tensor* t =
                ggml_new_tensor(vctx, GGML_TYPE_F32, ti->n_dims,
                                reinterpret_cast<const int64_t*>(ti->ne));
        ggml_set_name(t, name);
        staged_v.emplace_back(t, f);
        return t;
    };
    std::vector<ggml_tensor*> patch_embd = {vw("v.patch_embd.weight"),
                                            vw("v.patch_embd.weight.1")};
    ggml_tensor* patch_bias = vw("v.patch_embd.bias");
    ggml_tensor* pos_embd = vw("v.position_embd.weight");
    ggml_tensor* pre_ln_w = vw("v.pre_ln.weight");
    ggml_tensor* pre_ln_b = vw("v.pre_ln.bias");
    std::vector<int> layer_index(n_blocks);
    for (int32_t i = 0; i < n_blocks; ++i) layer_index[i] = i;
    std::vector<std::vector<ggml_tensor*>> vblocks;
    for (int32_t il = 0; il < n_blocks; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        std::vector<ggml_tensor*> blk(14);
        blk[0] = vw((p + "ln1.weight").c_str());
        blk[1] = vw((p + "ln1.bias").c_str());
        blk[2] = vw((p + "attn_qkv.weight").c_str());
        blk[3] = vw((p + "attn_qkv.bias").c_str());
        blk[4] = vw((p + "attn_out.weight").c_str());
        blk[5] = vw((p + "attn_out.bias").c_str());
        blk[6] = vw((p + "ln2.weight").c_str());
        blk[7] = vw((p + "ln2.bias").c_str());
        blk[8] = vw((p + "ffn_up.weight").c_str());
        blk[9] = vw((p + "ffn_up.bias").c_str());
        blk[10] = nullptr;  // qwen3vl vision FFN has no gate
        blk[11] = nullptr;
        blk[12] = vw((p + "ffn_down.weight").c_str());
        blk[13] = vw((p + "ffn_down.bias").c_str());
        vblocks.push_back(std::move(blk));
    }
    VisionTapsResult vres;
    std::string verr;
    ggml_cgraph* vg = build_vision_taps_graph(
            vctx, vp, tap_layers, tap_leaves, pixels, pos_leaf, patch_embd,
            patch_bias, pos_embd, pre_ln_w, pre_ln_b, layer_index, vblocks,
            &vres, &verr);
    if (vg == nullptr) {
        if (err != nullptr) *err = "vision taps graph: " + verr;
        ggml_free(vctx);
        return false;
    }

    // upload the pixel input, then execute
    ggml_gallocr_t galloc =
            ggml_gallocr_new(ggml_backend_dev_buffer_type(pixel_dev_));
    if (!ggml_gallocr_alloc_graph(galloc, vg)) {
        if (err != nullptr) *err = "vision graph alloc failed";
        ggml_gallocr_free(galloc);
        ggml_free(vctx);
        return false;
    }
    {
        // CHW plane layout == ggml (W, H, C) element order; weights staged
        // above are uploaded here too (no_alloc ctx: t->data is not writable
        // before gallocr).
        ggml_set_name(pixels, "pixels");
        ggml_set_name(pos_leaf, "pos_leaf");
        auto safe_set = [&](ggml_tensor* t, const void* data, size_t n) {
            if (t->buffer == nullptr) {
                WWV_LOG_WARN("tensor not in graph: %s", t->name);
                return;
            }
            ggml_backend_tensor_set(t, data, 0, n);
        };
        safe_set(pixels, vit.pixels.data(), ggml_nbytes(pixels));
        safe_set(pos_leaf, positions.data(), ggml_nbytes(pos_leaf));
        for (const auto& [t, host] : staged_v) {
            if (t->buffer == nullptr) {
                WWV_LOG_WARN("staged vision tensor not in graph: %s", t->name);
                continue;
            }
            ggml_backend_tensor_set(t, host, 0, ggml_nbytes(t));
        }
    }
    if (ggml_backend_graph_compute(pixel_backend_, vg) != GGML_STATUS_SUCCESS) {
        if (err != nullptr) *err = "vision graph compute failed";
        ggml_gallocr_free(galloc);
        ggml_free(vctx);
        return false;
    }
    const double perf_t3 = now_ms();  // vision graph computed
    // read taps back (before gallocr recycles the buffers): every tap is
    // dumped raw as a (C, T) flat for the torch-side bisect, then the
    // configured-layer taps are un-shuffled to the decoder's row-major maps
    // the configured feature layers are always the LAST `levels` taps
    // (sentinels -1/-2/0 only ever prepend)
    const size_t cfg_tap0 = tap_layers.size() - static_cast<size_t>(levels);
    {
        const int64_t C = n_embd_vit;
        for (size_t ti = 0; ti < tap_leaves.size(); ++ti) {
            // probe leaves may carry a different channel width (the qkv
            // sentinel is 3*C) — size the host buffer from the leaf itself
            std::vector<float> tap(ggml_nelements(tap_leaves[ti]));
            ggml_backend_tensor_get(tap_leaves[ti], tap.data(), 0,
                                    ggml_nbytes(tap_leaves[ti]));
            if (opts_.dump_dir.empty()) continue;
            const int64_t L = tap_layers[ti];
            std::string tag;
            if (L == -1)
                tag = "vit_patch_out";
            else if (L == -2)
                tag = "vit_pos_out";
            else if (L == -3)
                tag = "vit_b0_ln1";
            else if (L == -4)
                tag = "vit_b0_attn";
            else if (L == -5)
                tag = "vit_b0_ffn";
            else if (L == -6)
                tag = "vit_b0_qkv";
            else if (L == -7)
                tag = "vit_b0_qrope";
            else if (L == -8)
                tag = "vit_b0_v";
            else
                tag = "vit_blk" + std::to_string(L);
            std::FILE* f = std::fopen(
                    (opts_.dump_dir + "/" + tag + ".cpp.bin").c_str(), "wb");
            if (f != nullptr) {
                std::fwrite(tap.data(), 4, tap.size(), f);
                std::fclose(f);
                f = std::fopen((opts_.dump_dir + "/" + tag + ".shape").c_str(),
                               "w");
                if (f != nullptr) {
                    std::fprintf(f, "%lld,%lld",
                                 static_cast<long long>(tap_leaves[ti]->ne[0]),
                                 static_cast<long long>(n_tok));
                    std::fclose(f);
                }
            }
        }
        // host-side ViT inputs: the wwv-preprocessed pixel planes and the
        // m-rope position table (both consumed by the taps graph)
        std::FILE* f = std::fopen(
                (opts_.dump_dir + "/vit_pixels.cpp.bin").c_str(), "wb");
        if (f != nullptr) {
            std::fwrite(vit.pixels.data(), 4, vit.pixels.size(), f);
            std::fclose(f);
            f = std::fopen((opts_.dump_dir + "/vit_pixels.shape").c_str(), "w");
            if (f != nullptr) {
                std::fprintf(f, "3,%d,%d", vit.height, vit.width);
                std::fclose(f);
            }
        }
        f = std::fopen((opts_.dump_dir + "/vit_meta.txt").c_str(), "w");
        if (f != nullptr) {
            std::fprintf(f,
                         "w=%d h=%d patch=%d merge=%d npx=%lld npy=%lld "
                         "C=%lld heads=%d eps=%g n_tok=%lld\n",
                         vit.width, vit.height, patch, merge,
                         static_cast<long long>(npx),
                         static_cast<long long>(npy),
                         static_cast<long long>(n_embd_vit), heads, eps,
                         static_cast<long long>(n_tok));
            std::fclose(f);
        }
        f = std::fopen((opts_.dump_dir + "/vit_positions.cpp.bin").c_str(),
                       "wb");
        if (f != nullptr) {
            std::fwrite(positions.data(), 4, positions.size(), f);
            std::fclose(f);
            f = std::fopen((opts_.dump_dir + "/vit_positions.shape").c_str(),
                           "w");
            if (f != nullptr) {
                std::fprintf(f, "4,%lld", static_cast<long long>(n_tok));
                std::fclose(f);
            }
        }
    }
    for (int32_t li = 0; li < levels; ++li) {
        const int64_t C = n_embd_vit;
        const int64_t map_w = npx / merge;  // block grid
        const int64_t map_h = npy / merge;
        level_maps[li].assign(static_cast<size_t>(C) * n_tok, 0.0f);
    }
    {
        // The four tap readbacks + host un-shuffles are independent — run
        // them on parallel threads (the round-12 perf bisect showed the
        // serialized loop at ~310 ms; readback is a plain memcpy off the
        // graph buffer, safe to overlap).
        const int64_t C = n_embd_vit;
        const int64_t map_w = npx / merge;  // block grid
        const int64_t map_h = npy / merge;
        std::vector<std::thread> tap_workers;
        for (int32_t li = 0; li < levels; ++li) {
            auto work = [&, li] {
                std::vector<float> tap(static_cast<size_t>(C) * n_tok);
                ggml_backend_tensor_get(
                        tap_leaves[cfg_tap0 + static_cast<size_t>(li)],
                        tap.data(), 0,
                        ggml_nbytes(tap_leaves[cfg_tap0 +
                                               static_cast<size_t>(li)]));
                unshuffle_tokens_host(tap.data(), C, n_tok, map_w, map_h, merge,
                                      level_maps[li].data());
            };
            if (li + 1 < levels) {
                tap_workers.emplace_back(work);
            } else {
                work();  // the last layer runs inline
            }
        }
        for (auto& w : tap_workers) w.join();
    }
    ggml_gallocr_free(galloc);
    ggml_free(vctx);
    const double perf_t4 = now_ms();  // taps readback + unshuffle done

    // ---- 4. decoder graph (shared with the stages-in harness) ----
    if (!run_decoder_graph(rgb, std::move(cnn), std::move(level_maps),
                           std::move(t_seg), boxes, q, S, npx, npy, n_embd_vit,
                           n_embd, out, err)) {
        return false;
    }
    WWV_LOG_INFO(
            "[wwv-perf] pixel pass: preprocess %.0f ms | teacher %.0f ms | "
            "vit_graph %.0f ms | taps+unshuffle %.0f ms | decoder %.0f ms",
            perf_t1 - perf_t0, perf_t2 - perf_t1, perf_t3 - perf_t2,
            perf_t4 - perf_t3, now_ms() - perf_t4);
    return true;
}
bool WevisoneRuntime::run_decoder_graph(
        const RgbImage& rgb,
        std::vector<float>&& cnn,
        std::vector<std::vector<float>>&& level_maps,
        std::vector<float>&& t_seg,
        const std::vector<ResultBox>& boxes,
        int32_t q,
        int64_t S,
        int64_t npx,
        int64_t npy,
        int64_t n_embd_vit,
        int64_t n_embd,
        RunResult* out,
        std::string* err) {
    const int32_t levels = cfg_.msda_num_levels;
    const int64_t s1 = S;
    // ---- 4. decoder graph --------------------------------------------------
    ggml_context* dctx = nullptr;
    // CPU computes in-place on context-owned tensors (selftest-style:
    // no gallocr, no backend buffers) — this path is byte-exact and
    // side-steps any gallocr view-lifetime interaction. GPU keeps the
    // gallocr + backend path.
    const bool host_path = cpu_mode_;
    {
        const size_t estimate =
                host_path ? (65536ull + 512ull * q)
                                    << 20  // weights + inputs + all
                                           // intermediates, no reuse
                          : (128ull + 64ull * q) << 20;
        ggml_init_params ip{estimate, nullptr, !host_path};
        dctx = ggml_init(ip);
    }
    MaskDecoderInputs din;
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> staged_in;
    // Real-inference input snapshots: on the GPU path ggml_gallocr
    // recycles input-tensor memory during compute, so any dump read AFTER
    // the graph ran shows whatever the last consumer wrote there (the
    // round-10 mask investigation hit exactly this — the boxes01/cnn/
    // level dumps were garbage while the staged values were correct).
    // Snapshot the host buffers BEFORE they are staged/moved instead.
    std::map<std::string, std::vector<float>> input_snap;
    auto snap_input = [&](const std::string& name,
                          const std::vector<float>& v) {
        if (!opts_.dump_dir.empty()) input_snap[name] = v;
    };
    auto stage_input = [&](ggml_tensor* t, std::vector<float>&& host) {
        if (host_path) {
            std::memcpy(t->data, host.data(), host.size() * sizeof(float));
        } else {
            staged_in.emplace_back(t, std::move(host));
        }
        return t;
    };
    din.cnn_image = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, S, S, 3);
    snap_input("cnn_image", cnn);
    stage_input(din.cnn_image, std::move(cnn));
    for (int32_t l = 0; l < levels; ++l) {
        din.levels[l] =
                ggml_new_tensor_3d(dctx, GGML_TYPE_F32, npx, npy, n_embd_vit);
        snap_input("level" + std::to_string(l),
                   level_maps[static_cast<size_t>(l)]);
        stage_input(din.levels[l],
                    std::move(level_maps[static_cast<size_t>(l)]));
    }
    din.t_seg = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, n_embd, q);
    snap_input("t_seg", t_seg);
    stage_input(din.t_seg, std::move(t_seg));
    // boxes01: normalized xyxy — the stages-in harness feeds the torch
    // dump values directly; the normal path derives them from pixel boxes.
    std::vector<float> boxes01;
    std::vector<float> boxes01_host;  // outlives the builder call below
    if (!boxes01_direct_.empty()) {
        boxes01 = std::move(boxes01_direct_);
        boxes01_direct_.clear();
        boxes01_host = boxes01;
    } else {
        boxes01.assign(static_cast<size_t>(q) * 4, 0.0f);
        for (int32_t k = 0; k < q && k < static_cast<int32_t>(boxes.size());
             ++k) {
            boxes01[static_cast<size_t>(k) * 4 + 0] = std::clamp(
                    boxes[static_cast<size_t>(k)].x1 / std::max(1, rgb.width),
                    0.0f, 1.0f);
            boxes01[static_cast<size_t>(k) * 4 + 1] = std::clamp(
                    boxes[static_cast<size_t>(k)].y1 / std::max(1, rgb.height),
                    0.0f, 1.0f);
            boxes01[static_cast<size_t>(k) * 4 + 2] = std::clamp(
                    boxes[static_cast<size_t>(k)].x2 / std::max(1, rgb.width),
                    0.0f, 1.0f);
            boxes01[static_cast<size_t>(k) * 4 + 3] = std::clamp(
                    boxes[static_cast<size_t>(k)].y2 / std::max(1, rgb.height),
                    0.0f, 1.0f);
        }
        boxes01_host = boxes01;
    }
    din.boxes01 = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, 4, q);
    snap_input("boxes01", boxes01_host);
    stage_input(din.boxes01, std::move(boxes01));
    din.boxes01_host = boxes01_host.data();
    // stages-in MSDA isolation: torch query position replaces the in-graph
    // box-Fourier branch. torch dumps (q, d) row-major; ggml (d, q) — the
    // reversed-shape identity makes the raw bytes a straight memcpy.
    if (!query_position_direct_.empty()) {
        const size_t expect =
                static_cast<size_t>(q) * static_cast<size_t>(cfg_.hidden_dim);
        if (query_position_direct_.size() != expect) {
            if (err != nullptr) *err = "query_position override size mismatch";
            ggml_free(dctx);
            return false;
        }
        din.query_position =
                ggml_new_tensor_2d(dctx, GGML_TYPE_F32, cfg_.hidden_dim, q);
        stage_input(din.query_position, std::move(query_position_direct_));
        query_position_direct_.clear();
    }

    // weights from the decoder GGUF -> graph leaves
    std::vector<std::pair<ggml_tensor*, const float*>> staged_d;
    auto dw = [&](const char* name) -> ggml_tensor* {
        const auto* ti = decoder_gguf_->tensor(name);
        if (ti == nullptr) return nullptr;
        const float* f = decoder_gguf_->tensor_f32(name);
        if (f == nullptr) return nullptr;
        ggml_tensor* t =
                ggml_new_tensor(dctx, GGML_TYPE_F32, ti->n_dims,
                                reinterpret_cast<const int64_t*>(ti->ne));
        ggml_set_name(t, name);
        if (host_path) {
            std::memcpy(t->data, f, ggml_nbytes(t));
        } else {
            staged_d.emplace_back(t, f);
        }
        return t;
    };
    auto dwb = [&](const std::string& n) { return dw(n.c_str()); };
    MaskDecoderWeights W;
    std::string missing;
    auto need = [&](ggml_tensor* t, const char* what) {
        if (t == nullptr) missing += std::string(what) + "; ";
        return t;
    };
    for (int32_t l = 0; l < levels; ++l) {
        W.input_proj_w[l] = need(dwb("visual_encoder.input_proj." +
                                     std::to_string(l) + ".0.weight"),
                                 "input_proj_w");
        W.input_proj_gn_w[l] = dwb("visual_encoder.input_proj." +
                                   std::to_string(l) + ".1.weight");
        W.input_proj_gn_b[l] = dwb("visual_encoder.input_proj." +
                                   std::to_string(l) + ".1.bias");
    }
    W.level_embed = need(dwb("visual_encoder.level_embed"), "level_embed");
    const int32_t enc_layers = cfg_.msda_encoder_layers;
    for (int32_t li = 0; li < enc_layers; ++li) {
        const std::string p =
                "visual_encoder.encoder.layers." + std::to_string(li) + ".";
        std::array<ggml_tensor*, 10> lw{};
        lw[0] = need(dwb(p + "depthwise.weight"), "enc depthwise");
        lw[1] = need(dwb(p + "pointwise.weight"), "enc pointwise");
        lw[2] = dwb(p + "norm1.weight");
        lw[3] = dwb(p + "norm1.bias");
        lw[4] = need(dwb(p + "linear1.weight"), "enc linear1");
        lw[5] = dwb(p + "linear1.bias");
        lw[6] = need(dwb(p + "linear2.weight"), "enc linear2");
        lw[7] = dwb(p + "linear2.bias");
        lw[8] = dwb(p + "norm2.weight");
        lw[9] = dwb(p + "norm2.bias");
        W.encoder_layers.push_back(lw);
    }
    W.content_proj_w =
            need(dwb("query_builder.content_proj.weight"), "content_proj");
    W.content_proj_b = dwb("query_builder.content_proj.bias");
    W.content_norm_w = dwb("query_builder.content_norm.weight");
    W.content_norm_b = dwb("query_builder.content_norm.bias");
    W.box_mlp_w1 =
            need(dwb("query_builder.box_position.mlp.0.weight"), "box mlp1");
    W.box_mlp_b1 = dwb("query_builder.box_position.mlp.0.bias");
    W.box_mlp_w2 =
            need(dwb("query_builder.box_position.mlp.2.weight"), "box mlp2");
    W.box_mlp_b2 = dwb("query_builder.box_position.mlp.2.bias");
    W.position_norm_w = dwb("query_builder.position_norm.weight");
    W.position_norm_b = dwb("query_builder.position_norm.bias");
    // host constant: 2^arange(num_fourier_freqs)
    {
        ggml_tensor* bf = ggml_new_tensor_2d(dctx, GGML_TYPE_F32,
                                             cfg_.num_fourier_freqs, 1);
        std::vector<float> d(cfg_.num_fourier_freqs);
        for (int32_t i = 0; i < cfg_.num_fourier_freqs; ++i) {
            d[static_cast<size_t>(i)] = std::pow(2.0f, static_cast<float>(i));
        }
        stage_input(bf, std::move(d));
        W.box_freqs = bf;
    }
    for (int32_t li = 0; li < cfg_.transformer_layers; ++li) {
        const std::string p = "transformer.layers." + std::to_string(li) + ".";
        std::array<ggml_tensor*, 20> tw{};
        tw[0] = need(dwb(p + "self_attn.in_proj_weight"), "self in_proj");
        tw[1] = dwb(p + "self_attn.in_proj_bias");
        tw[2] = need(dwb(p + "self_attn.out_proj.weight"), "self out");
        tw[3] = dwb(p + "self_attn.out_proj.bias");
        tw[4] = dwb(p + "norm1.weight");
        tw[5] = dwb(p + "norm1.bias");
        tw[6] = need(dwb(p + "cross_attn.value_proj.weight"), "msda value");
        tw[7] = dwb(p + "cross_attn.value_proj.bias");
        tw[8] = need(dwb(p + "cross_attn.sampling_offsets.weight"), "msda off");
        tw[9] = dwb(p + "cross_attn.sampling_offsets.bias");
        tw[10] = need(dwb(p + "cross_attn.attention_weights.weight"),
                      "msda att");
        tw[11] = dwb(p + "cross_attn.attention_weights.bias");
        tw[12] = need(dwb(p + "cross_attn.output_proj.weight"), "msda out");
        tw[13] = dwb(p + "cross_attn.output_proj.bias");
        tw[14] = dwb(p + "norm2.weight");
        tw[15] = dwb(p + "norm2.bias");
        tw[16] = need(dwb(p + "linear1.weight"), "ffn1");
        tw[17] = dwb(p + "linear1.bias");
        tw[18] = need(dwb(p + "linear2.weight"), "ffn2");
        tw[19] = dwb(p + "linear2.bias");
        W.transformer_layers.push_back(tw);
        W.transformer_norms.emplace_back(dwb(p + "norm3.weight"),
                                         dwb(p + "norm3.bias"));
    }
    for (int32_t l = 0; l < levels; ++l) {
        W.fpn_lateral_w[l] = need(dwb("pixel_fusion.fpn.lateral." +
                                      std::to_string(l) + ".weight"),
                                  "fpn lateral");
    }
    for (int32_t i = 0; i < levels - 1; ++i) {
        const std::string p =
                "pixel_fusion.fpn.output." + std::to_string(i) + ".";
        std::array<ggml_tensor*, 6> ow{};
        ow[0] = need(dwb(p + "0.weight"), "fpn out conv");
        ow[1] = need(dwb(p + "1.weight"), "fpn out gn");
        ow[2] = dwb(p + "1.bias");
        W.fpn_output.push_back(ow);
    }
    W.up_conv0_w = need(dwb("pixel_fusion.upsample.0.conv.weight"), "up0");
    W.up_conv0_b = dwb("pixel_fusion.upsample.0.conv.bias");
    W.up_conv1_w = need(dwb("pixel_fusion.upsample.1.conv.weight"), "up1");
    W.up_conv1_b = dwb("pixel_fusion.upsample.1.conv.bias");
    W.s2_up_conv_w = need(dwb("pixel_fusion.s2_upsample.conv.weight"), "s2up");
    W.s2_up_conv_b = dwb("pixel_fusion.s2_upsample.conv.bias");
    W.stem_conv0_w = need(dwb("pixel_fusion.cnn.net.0.weight"), "stem0");
    W.stem_gn0_w = dwb("pixel_fusion.cnn.net.1.weight");
    W.stem_gn0_b = dwb("pixel_fusion.cnn.net.1.bias");
    W.stem_conv1_w = need(dwb("pixel_fusion.cnn.net.3.weight"), "stem1");
    W.stem_gn1_w = dwb("pixel_fusion.cnn.net.4.weight");
    W.stem_gn1_b = dwb("pixel_fusion.cnn.net.4.bias");
    W.fpn_proj_w = need(dwb("pixel_fusion.fpn_proj.weight"), "fpn proj");
    W.cnn_proj_w = need(dwb("pixel_fusion.cnn_proj.weight"), "cnn proj");
    W.smooth_gn_w = need(dwb("pixel_fusion.smooth.0.weight"), "smooth gn");
    W.smooth_gn_b = dwb("pixel_fusion.smooth.0.bias");
    W.smooth_conv_w = need(dwb("pixel_fusion.smooth.2.weight"), "smooth conv");
    W.kernel_w1 = need(dwb("mask_predictor.kernel.0.weight"), "kernel1");
    W.kernel_b1 = dwb("mask_predictor.kernel.0.bias");
    W.kernel_w2 = need(dwb("mask_predictor.kernel.2.weight"), "kernel2");
    W.kernel_b2 = dwb("mask_predictor.kernel.2.bias");
    const std::string hr = "highres_refine.";
    W.cascade_semantic_w =
            need(dwb(hr + "semantic.0.weight"), "cascade semantic");
    W.cascade_semantic_gn_w = dwb(hr + "semantic.1.weight");
    W.cascade_semantic_gn_b = dwb(hr + "semantic.1.bias");
    W.rgb_s4_w = need(dwb(hr + "rgb_s4.0.weight"), "rgb_s4");
    W.rgb_s4_gn_w = dwb(hr + "rgb_s4.1.weight");
    W.rgb_s4_gn_b = dwb(hr + "rgb_s4.1.bias");
    W.rgb_s2_w = need(dwb(hr + "rgb_s2.0.weight"), "rgb_s2");
    W.rgb_s2_gn_w = dwb(hr + "rgb_s2.1.weight");
    W.rgb_s2_gn_b = dwb(hr + "rgb_s2.1.bias");
    W.rgb_s1_w = need(dwb(hr + "rgb_s1.0.weight"), "rgb_s1");
    W.rgb_s1_gn_w = dwb(hr + "rgb_s1.1.weight");
    W.rgb_s1_gn_b = dwb(hr + "rgb_s1.1.bias");
    W.coarse_s4_w = need(dwb(hr + "coarse_s4.0.weight"), "coarse_s4");
    W.coarse_s4_gn_w = dwb(hr + "coarse_s4.1.weight");
    W.coarse_s4_gn_b = dwb(hr + "coarse_s4.1.bias");
    W.coarse_s2_w = need(dwb(hr + "coarse_s2.0.weight"), "coarse_s2");
    W.coarse_s2_gn_w = dwb(hr + "coarse_s2.1.weight");
    W.coarse_s2_gn_b = dwb(hr + "coarse_s2.1.bias");
    W.previous_s2_w = need(dwb(hr + "previous_s2.0.weight"), "prev_s2");
    W.previous_s2_gn_w = dwb(hr + "previous_s2.1.weight");
    W.previous_s2_gn_b = dwb(hr + "previous_s2.1.bias");
    W.previous_s1_w = need(dwb(hr + "previous_s1.0.weight"), "prev_s1");
    W.previous_s1_gn_w = dwb(hr + "previous_s1.1.weight");
    W.previous_s1_gn_b = dwb(hr + "previous_s1.1.bias");
    W.fuse_s4_w = need(dwb(hr + "fuse_s4.weight"), "fuse_s4");
    W.fuse_s2_w = need(dwb(hr + "fuse_s2.weight"), "fuse_s2");
    W.fuse_s1_w = need(dwb(hr + "fuse_s1.weight"), "fuse_s1");
    W.up_s4_s2_w = need(dwb(hr + "up_s4_to_s2.weight"), "convT s4s2");
    W.up_s2_s1_w = need(dwb(hr + "up_s2_to_s1.weight"), "convT s2s1");
    W.query_s4_w = need(dwb(hr + "query_s4.weight"), "query_s4");
    W.query_s4_b = dwb(hr + "query_s4.bias");
    W.query_s2_w = need(dwb(hr + "query_s2.weight"), "query_s2");
    W.query_s2_b = dwb(hr + "query_s2.bias");
    W.query_s1_w = need(dwb(hr + "query_s1.weight"), "query_s1");
    W.query_s1_b = dwb(hr + "query_s1.bias");
    W.output_s4_w = need(dwb(hr + "output_s4.weight"), "output_s4");
    W.output_s4_b = dwb(hr + "output_s4.bias");
    W.output_s2_w = need(dwb(hr + "output_s2.weight"), "output_s2");
    W.output_s2_b = dwb(hr + "output_s2.bias");
    W.output_s1_w = need(dwb(hr + "output_s1.weight"), "output_s1");
    W.output_s1_b = dwb(hr + "output_s1.bias");
    const std::vector<std::string> stack_names = {"blocks_s4", "blocks_s2",
                                                  "blocks_s1"};
    for (int32_t st = 0; st < 3; ++st) {
        std::array<ggml_tensor*, 12> stack{};
        for (int32_t b = 0; b < cfg_.alpha_cascade_blocks; ++b) {
            const std::string p = hr + stack_names[static_cast<size_t>(st)] +
                                  "." + std::to_string(b) + ".net.";
            stack[static_cast<size_t>(b) * 6 + 0] =
                    need(dwb(p + "0.weight"), "res gn");
            stack[static_cast<size_t>(b) * 6 + 1] = dwb(p + "0.bias");
            stack[static_cast<size_t>(b) * 6 + 2] =
                    need(dwb(p + "2.weight"), "res conv");
            stack[static_cast<size_t>(b) * 6 + 3] =
                    need(dwb(p + "3.weight"), "res gn2");
            stack[static_cast<size_t>(b) * 6 + 4] = dwb(p + "3.bias");
            stack[static_cast<size_t>(b) * 6 + 5] =
                    need(dwb(p + "5.weight"), "res conv2");
        }
        W.cascade_blocks.push_back(stack);
    }
    if (!missing.empty()) {
        if (err != nullptr) *err = "decoder tensors missing: " + missing;
        ggml_free(dctx);
        return false;
    }
    // host constant: sine 2-D position embedding for the encoder levels
    {
        W.encoder_pos_leaf = ggml_new_tensor_2d(
                dctx, GGML_TYPE_F32, cfg_.hidden_dim, levels * npx * npy);
        stage_input(
                W.encoder_pos_leaf,
                build_sine_pos_leaf(cfg_.hidden_dim, static_cast<int32_t>(npx),
                                    static_cast<int32_t>(npy), levels));
    }

    WWV_LOG_DEBUG("ctx used after staging: %.2f GB", ggml_used_mem(dctx) / 1e9);
    // The parity harness may run at a reduced cnn canvas: the builder
    // derives s2/s4/s1 from cfg.cnn_image_size, so it must follow S.
    wwv::DecoderConfig cfg_eff = cfg_;
    cfg_eff.cnn_image_size = static_cast<int32_t>(S);
    MaskDecoderOutputs dout;
    std::string derr;
    std::vector<std::pair<std::string, ggml_tensor*>> taps;
    const bool want_taps = !opts_.dump_dir.empty() && !opts_.no_taps;
    ggml_cgraph* dg = build_mask_decoder_graph(
            dctx, cfg_eff, q, static_cast<int32_t>(npx),
            static_cast<int32_t>(npy), n_embd_vit, n_embd, W, din, &dout, &derr,
            want_taps ? &taps : nullptr,
            opts_.taps_filter.empty() ? nullptr : opts_.taps_filter.c_str(),
            // host compute path: op outputs stay valid in the non-recycling
            // pool after graph_compute — tap them directly (no cpy nodes, whose
            // insertion perturbs execution).
            host_path);
    if (dg == nullptr) {
        if (err != nullptr) *err = "decoder graph: " + derr;
        ggml_free(dctx);
        return false;
    }
    if (host_path) {
        // graph sanity: are the gather nodes present and in dependency order?
        int n_gr = 0, n_cpy = 0;
        const int nn = ggml_graph_n_nodes(dg);
        for (int i = 0; i < nn; ++i) {
            ggml_tensor* nd = ggml_graph_node(dg, i);
            if (nd->op == GGML_OP_GET_ROWS) ++n_gr;
            // this pinned ggml implements cast as a type-changing GGML_OP_CPY
            if (nd->op == GGML_OP_CPY) ++n_cpy;
        }
        WWV_LOG_DEBUG("graph ops: %d get_rows, %d cpy/cast of %d nodes", n_gr,
                      n_cpy, nn);
        size_t total = 0;
        for (int i = 0; i < nn; ++i) {
            total += ggml_nbytes(ggml_graph_node(dg, i));
        }
        WWV_LOG_DEBUG("graph data pool need: %.1f GB (%d nodes)", total / 1e9,
                      nn);
    }

    ggml_gallocr_t dgalloc = nullptr;
    if (!host_path) {
        dgalloc = ggml_gallocr_new(ggml_backend_dev_buffer_type(pixel_dev_));
        if (!ggml_gallocr_alloc_graph(dgalloc, dg)) {
            if (err != nullptr)
                *err = "decoder graph alloc failed (device memory)";
            ggml_gallocr_free(dgalloc);
            ggml_free(dctx);
            return false;
        }
    }
    // upload every staged host buffer (weights + inputs) after gallocr
    for (const auto& [t, host] : staged_in) {
        if (t->buffer == nullptr) {
            WWV_LOG_WARN("staged input not in graph: %s", t->name);
            continue;
        }
        ggml_backend_tensor_set(t, host.data(), 0, host.size() * sizeof(float));
    }
    for (const auto& [t, host] : staged_d) {
        if (t->buffer == nullptr) {
            WWV_LOG_WARN("staged weight not in graph: %s", t->name);
            continue;
        }
        ggml_backend_tensor_set(t, host, 0, ggml_nbytes(t));
    }
    for (const auto& [t, v] : wwv_f32_leaves()) {
        if (host_path) {
            // no_alloc=false: the context owns the bytes, write in place.
            std::memcpy(t->data, &v, sizeof(float));
            continue;
        }
        if (t->buffer == nullptr) {
            WWV_LOG_WARN("f32 leaf not in graph");
            continue;
        }
        ggml_backend_tensor_set(t, &v, 0, sizeof(float));
    }
    for (const auto& [t, tab] : wwv_perm_staged()) {
        if (host_path) {
            std::memcpy(t->data, tab.data(), tab.size() * sizeof(int32_t));
            continue;
        }
        if (t->buffer == nullptr) {
            WWV_LOG_WARN("perm table not in graph");
            continue;
        }
        ggml_backend_tensor_set(t, tab.data(), 0, tab.size() * sizeof(int32_t));
    }
    for (const auto& [t, v] : wwv_f32vec_staged()) {
        if (host_path) {
            std::memcpy(t->data, v.data(), v.size() * sizeof(float));
            continue;
        }
        if (t->buffer == nullptr) {
            WWV_LOG_WARN("f32 vec leaf not in graph");
            continue;
        }
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    }
    if (host_path) {
        // selftest-style compute: the context owns host memory, the graph
        // runs in place (no gallocr, no backend buffers). The plan/compute
        // pair lives in the dlopen'd CPU backend module (GGML_BACKEND_DL
        // builds have no import library), reachable only through the
        // official registry extension point (the sam3/rmbg pattern) —
        // calling them directly would leave undefined symbols in
        // libAICore.so.
        using PlanFn =
                ggml_cplan (*)(const ggml_cgraph*, int, ggml_threadpool_t);
        using ComputeFn = ggml_status (*)(ggml_cgraph*, ggml_cplan*);
        ggml_backend_dev_t dev = ggml_backend_get_device(pixel_backend_);
        ggml_backend_reg_t reg =
                dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
        auto plan_fn = reg != nullptr
                               ? reinterpret_cast<PlanFn>(
                                         ggml_backend_reg_get_proc_address(
                                                 reg, "ggml_graph_plan"))
                               : nullptr;
        auto compute_fn = reg != nullptr
                                  ? reinterpret_cast<ComputeFn>(
                                            ggml_backend_reg_get_proc_address(
                                                    reg, "ggml_graph_compute"))
                                  : nullptr;
        if (plan_fn == nullptr || compute_fn == nullptr) {
            if (err != nullptr)
                *err = "CPU backend module lacks ggml_graph_plan/compute";
            ggml_free(dctx);
            return false;
        }
        ggml_cplan cplan = plan_fn(dg, threads_, nullptr);
        std::vector<uint8_t> work(cplan.work_size + 64, 0);
        cplan.work_data = work.data();
        if (compute_fn(dg, &cplan) != GGML_STATUS_SUCCESS) {
            if (err != nullptr) *err = "decoder graph compute failed";
            ggml_free(dctx);
            return false;
        }
    } else {
        if (ggml_backend_graph_compute(pixel_backend_, dg) !=
            GGML_STATUS_SUCCESS) {
            if (err != nullptr) *err = "decoder graph compute failed";
            ggml_gallocr_free(dgalloc);
            ggml_free(dctx);
            return false;
        }
    }
    // stage-boundary tap dumps for the torch-side bisect
    if (!opts_.dump_dir.empty()) {
        for (const auto& [name, t] : taps) {
            if (!host_path && t->buffer == nullptr) {
                WWV_LOG_WARN("tap not in graph: %s", name.c_str());
                continue;
            }
            // Reversed-shape identity: the ggml flat IS the torch flat
            // whenever ne == reversed(torch shape) (e.g. (C,T) vs (T,C),
            // (W,H,C) vs (C,H,W)). Every interleave here corrupted dumps;
            // taps and stage dumps are raw memcpys only.
            std::vector<float> host(ggml_nelements(t));
            if (host_path) {
                std::memcpy(host.data(), t->data, host.size() * 4);
            } else {
                ggml_backend_tensor_get(t, host.data(), 0, host.size() * 4);
            }
            const float* outv = host.data();
            const std::string base = opts_.dump_dir + "/tap_" + name;
            std::string sh;
            if (name.rfind("enc_w", 0) == 0) {
                sh = std::to_string(t->ne[0]) + "," + std::to_string(t->ne[1]);
            } else if (name == "pos_leaf") {
                // (d, levels*hw) raw; torch layout per level (C, H, W)
                sh = std::to_string(t->ne[0]) + "," + std::to_string(npy) +
                     "," + std::to_string(npx) + "," + std::to_string(4);
            } else if (name.rfind("enc_proj", 0) == 0) {
                // tapped tensor is the conv1x1 output (W, H, C): torch
                // (C, H, W) flat via the reversed-shape identity
                sh = std::to_string(t->ne[2]) + "," + std::to_string(npy) +
                     "," + std::to_string(npx);
            }
            std::FILE* f = std::fopen((base + ".cpp.bin").c_str(), "wb");
            if (f != nullptr) {
                std::fwrite(outv, 4, host.size(), f);
                std::fclose(f);
            }
            f = std::fopen((base + ".shape").c_str(), "w");
            if (f != nullptr) {
                if (!sh.empty()) {
                    std::fputs(sh.c_str(), f);
                } else if (name.rfind("enc_conv", 0) == 0 ||
                           name == "pixel_fusion") {
                    // (W, H, C, 1) graph layout -> torch (C, H, W)
                    std::fprintf(f, "%lld,%lld,%lld",
                                 static_cast<long long>(t->ne[2]),
                                 static_cast<long long>(t->ne[1]),
                                 static_cast<long long>(t->ne[0]));
                } else {
                    // generic 2-D: rows = ne[1], cols = ne[0]
                    std::fprintf(f, "%lld,%lld",
                                 static_cast<long long>(t->ne[1]),
                                 static_cast<long long>(t->ne[0]));
                }
                std::fclose(f);
            }
        }
    }

    // ---- parity dump (when opts_.dump_dir is set) ----
    if (!opts_.dump_dir.empty()) {
        auto dump_tensor = [&](const char* name, ggml_tensor* t,
                               const char* shape) {
            std::vector<float> host;
            const float* src;
            size_t n;
            const auto snap = input_snap.find(name);
            if (snap != input_snap.end()) {
                // real-inference input: read the pre-compute snapshot —
                // the tensor memory was recycled by gallocr
                src = snap->second.data();
                n = snap->second.size();
            } else {
                host.resize(ggml_nelements(t));
                if (host_path) {
                    std::memcpy(host.data(), t->data, host.size() * 4);
                } else {
                    ggml_backend_tensor_get(t, host.data(), 0, host.size() * 4);
                }
                src = host.data();
                n = host.size();
            }
            const std::string base = opts_.dump_dir + "/" + name;
            std::FILE* f = std::fopen((base + ".cpp.bin").c_str(), "wb");
            if (f == nullptr) return;
            std::fwrite(src, 4, n, f);
            std::fclose(f);
            f = std::fopen((base + ".shape").c_str(), "w");
            if (f != nullptr) {
                std::fputs(shape, f);
                std::fclose(f);
            }
        };
        const std::string cshape = std::to_string(3) + "," + std::to_string(S) +
                                   "," + std::to_string(S);
        dump_tensor("cnn_image", din.cnn_image, cshape.c_str());
        for (int32_t l = 0; l < levels; ++l) {
            const std::string lshape = std::to_string(n_embd_vit) + "," +
                                       std::to_string(npy) + "," +
                                       std::to_string(npx);
            dump_tensor(("level" + std::to_string(l)).c_str(), din.levels[l],
                        lshape.c_str());
        }
        dump_tensor("t_seg", din.t_seg,
                    (std::to_string(q) + "," + std::to_string(n_embd)).c_str());
        dump_tensor("boxes01", din.boxes01, (std::to_string(q) + ",4").c_str());
        // per-query planes: (W, H, 1, Q) -> (Q, H, W) row-major
        auto dump_planes = [&](const char* name, ggml_tensor* t) {
            const int64_t w = t->ne[0];
            const int64_t h = t->ne[1];
            const int64_t n = t->ne[3] > 0 ? t->ne[3] : 1;
            std::vector<float> host(ggml_nelements(t));
            if (host_path) {
                std::memcpy(host.data(), t->data, host.size() * 4);
            } else {
                ggml_backend_tensor_get(t, host.data(), 0, host.size() * 4);
            }
            // The (W, H, 1, q) ggml flat (x + W*y + W*H*q) IS the torch
            // (q, H, W) flat (q*H*W + y*W + x): the trailing query axis is
            // the slowest one on both sides — reversed-shape identity.
            // Any interleave here CORRUPTS the dump.
            const float* outv = host.data();
            const std::string base = opts_.dump_dir + "/" + name;
            std::FILE* f = std::fopen((base + ".cpp.bin").c_str(), "wb");
            if (f != nullptr) {
                std::fwrite(outv, 4, host.size(), f);
                std::fclose(f);
            }
            f = std::fopen((base + ".shape").c_str(), "w");
            if (f != nullptr) {
                const std::string sh = std::to_string(n) + "," +
                                       std::to_string(h) + "," +
                                       std::to_string(w);
                std::fputs(sh.c_str(), f);
                std::fclose(f);
            }
        };
        if (dout.mask_logits_coarse != nullptr) {
            dump_planes("coarse", dout.mask_logits_coarse);
        }
        if (dout.mask_logits_s2 != nullptr) {
            dump_planes("logits_s2", dout.mask_logits_s2);
        }
        if (dout.mask_logits_s1 != nullptr) {
            dump_planes("logits_s1", dout.mask_logits_s1);
        }
    }

    // ---- 5. matte extraction (first query; parity-first scope) ----
    out->matte_width = static_cast<int32_t>(s1);
    out->matte_height = static_cast<int32_t>(s1);
    out->matte.assign(static_cast<size_t>(s1) * s1, 0.0f);
    {
        ggml_tensor* logits = dout.mask_logits_s1;  // (S, S, 1, q)
        std::vector<float> raw(static_cast<size_t>(s1) * s1);
        const size_t plane = static_cast<size_t>(s1) * s1;
        if (host_path) {
            std::memcpy(raw.data(), logits->data, plane * 4);
        } else {
            ggml_backend_tensor_get(logits, raw.data(), 0, plane * 4);
        }
        const int32_t ow = rgb.width;
        const int32_t oh = rgb.height;
        out->matte.assign(static_cast<size_t>(ow) * oh, 0.0f);
        for (int32_t y = 0; y < oh; ++y) {
            const float fy = (y + 0.5f) * static_cast<float>(s1) / oh - 0.5f;
            const int32_t y0 = std::clamp(static_cast<int32_t>(std::floor(fy)),
                                          0, static_cast<int32_t>(s1 - 1));
            const int32_t y1t =
                    std::clamp(y0 + 1, 0, static_cast<int32_t>(s1 - 1));
            const float wy = std::clamp(fy - y0, 0.0f, 1.0f);
            for (int32_t x = 0; x < ow; ++x) {
                const float fx =
                        (x + 0.5f) * static_cast<float>(s1) / ow - 0.5f;
                const int32_t x0 =
                        std::clamp(static_cast<int32_t>(std::floor(fx)), 0,
                                   static_cast<int32_t>(s1 - 1));
                const int32_t x1t =
                        std::clamp(x0 + 1, 0, static_cast<int32_t>(s1 - 1));
                const float wx = std::clamp(fx - x0, 0.0f, 1.0f);
                const float a00 =
                        1.0f /
                        (1.0f +
                         std::exp(-raw[static_cast<size_t>(y0) * s1 + x0]));
                const float a10 =
                        1.0f /
                        (1.0f +
                         std::exp(-raw[static_cast<size_t>(y0) * s1 + x1t]));
                const float a01 =
                        1.0f /
                        (1.0f +
                         std::exp(-raw[static_cast<size_t>(y1t) * s1 + x0]));
                const float a11 =
                        1.0f /
                        (1.0f +
                         std::exp(-raw[static_cast<size_t>(y1t) * s1 + x1t]));
                out->matte[static_cast<size_t>(y) * ow + x] =
                        a00 * (1 - wx) * (1 - wy) + a10 * wx * (1 - wy) +
                        a01 * (1 - wx) * wy + a11 * wx * wy;
            }
        }
    }
    out->masks_decoded = true;
    out->matte_width = rgb.width;
    out->matte_height = rgb.height;
    ggml_gallocr_free(dgalloc);
    ggml_free(dctx);
    return true;
}
}  // namespace wwv
