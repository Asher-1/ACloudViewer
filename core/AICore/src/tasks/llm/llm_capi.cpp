// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Local LLM / VLM generation over the pinned llama.cpp (libllama + libmtmd).
// llama.cpp is its own upstream here: this file only maps the AICore
// contract (options, borrowed image views, typed results, timings, cancel)
// onto the public llama/mtmd C API, so greedy output is comparable token for
// token with the upstream tools on the same revision, model and prompt.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "aicore/llm_capi.h"
#include "aicore/runtime_capi.h"
#include "common/data_root_util.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/image_utils.hpp"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// llama_backend_init only initializes process-wide timing/quantization
// tables, and llama_log_set also replaces ggml's logger: both are global and
// therefore done once per process, never per context.
void init_llama_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_common::load_backends_once();
        llama_backend_init();
        llama_log_set(ggml_common::log_bridge_callback(), nullptr);
        mtmd_helper_log_set(ggml_common::log_bridge_callback(), nullptr);
    });
}

// Device request -> registry device. "auto"/"gpu" walk the build's backend
// priority and fall back to CPU; an explicit family/index that is absent is
// a typed error, never a silent CPU run. nullptr means CPU.
ggml_backend_dev_t resolve_device(const std::string& request) {
    std::string family;
    int index = 0;
    ggml_common::parse_device(request.empty() ? "auto" : request, family,
                              index);
    if (family == "cpu") return nullptr;
    const bool any = family == "auto" || family == "gpu";
    auto nth_gpu = [](const std::string& want, int want_index) {
        int seen = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
                type != GGML_BACKEND_DEVICE_TYPE_IGPU)
                continue;
            const char* reg =
                    ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
            if (ggml_common::registry_backend_id(reg) != want) continue;
            if (seen++ == want_index) return dev;
        }
        return ggml_backend_dev_t(nullptr);
    };
    if (any) {
        for (const char* const* p = ggml_common::auto_backend_ids(); *p; ++p)
            if (auto* dev = nth_gpu(*p, index)) return dev;
        return nullptr;
    }
    if (auto* dev = nth_gpu(ggml_common::registry_backend_id(family.c_str()),
                            index))
        return dev;
    throw std::runtime_error("requested device '" + request +
                             "' is not available");
}

size_t count_occurrences(const std::string& text, const std::string& needle) {
    size_t count = 0;
    for (size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + needle.size()))
        ++count;
    return count;
}

// Largest prefix length <= end of `text` that ends on a UTF-8 code point
// boundary (an incomplete trailing sequence is held back for streaming).
size_t complete_utf8_prefix(const std::string& text, size_t end) {
    if (end == 0) return 0;
    size_t lead = end - 1;
    while (lead > 0 && end - lead < 4 &&
           (static_cast<unsigned char>(text[lead]) & 0xC0) == 0x80)
        --lead;
    const unsigned char ch = static_cast<unsigned char>(text[lead]);
    const size_t need = ch < 0x80                ? 1
                        : (ch & 0xE0) == 0xC0 ? 2
                        : (ch & 0xF0) == 0xE0 ? 3
                        : (ch & 0xF8) == 0xF0 ? 4
                                              : 1;  // invalid: pass through
    return end - lead >= need ? end : lead;
}

struct OwnedResult {
    aicore_llm_result public_result{};  // first member, released by address
    std::string text;
    std::vector<int32_t> tokens;
};

template <class T>
bool valid(const T* value) {
    return value && value->struct_size >= sizeof(T) &&
           value->abi_version == AICORE_LLM_ABI_VERSION;
}

struct SamplerDeleter {
    void operator()(llama_sampler* s) const { llama_sampler_free(s); }
};
struct ChunksDeleter {
    void operator()(mtmd_input_chunks* c) const { mtmd_input_chunks_free(c); }
};
struct BitmapDeleter {
    void operator()(mtmd_bitmap* b) const { mtmd_bitmap_free(b); }
};

}  // namespace

struct aicore_llm_ctx {
    llama_model* model = nullptr;
    llama_context* lctx = nullptr;
    mtmd_context* mctx = nullptr;
    std::string device = "CPU";
    std::string error;
    std::atomic<bool> cancel{false};
    aicore_pipeline_timings timings{};
    bool has_timings = false;

    ~aicore_llm_ctx() {
        if (mctx) mtmd_free(mctx);
        if (lctx) llama_free(lctx);
        if (model) llama_model_free(model);
    }
    bool ready() const { return model && lctx; }
    bool cancelled() const {
        return cancel.load(std::memory_order_relaxed) ||
               aicore_cancel_requested();
    }
};

namespace {

bool abort_callback(void* data) {
    return static_cast<const aicore_llm_ctx*>(data)->cancelled();
}

void load(aicore_llm_ctx& ctx, const aicore_llm_options& options) {
    if (!options.model_path || !*options.model_path)
        throw std::runtime_error("model_path is required");
    if (options.flash_attn < -1 || options.flash_attn > 1)
        throw std::runtime_error("flash_attn must be -1, 0 or 1");
    if (options.image_max_tokens < 0)
        throw std::runtime_error("image_max_tokens must be >= 0");
    init_llama_once();
    ggml_backend_dev_t dev =
            resolve_device(options.device ? options.device : "auto");
    ggml_backend_dev_t devices[2] = {dev, nullptr};
    ctx.device = dev ? ggml_backend_dev_name(dev) : "CPU";

    llama_model_params mp = llama_model_default_params();
    mp.devices = devices;  // {nullptr} => no accelerator, CPU only
    mp.n_gpu_layers = dev ? -1 : 0;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu = 0;
    ctx.model = llama_model_load_from_file(options.model_path, mp);
    if (!ctx.model)
        throw std::runtime_error(std::string("failed to load model: ") +
                                 options.model_path);

    const int threads = options.threads > 0
                                ? options.threads
                                : int(ggml_common::default_cpu_threads());
    const auto flash = options.flash_attn < 0 ? LLAMA_FLASH_ATTN_TYPE_AUTO
                       : options.flash_attn == 0
                               ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                               : LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_params cp = llama_context_default_params();
    const int32_t train = llama_model_n_ctx_train(ctx.model);
    cp.n_ctx = options.context_tokens
                       ? options.context_tokens
                       : uint32_t(std::min<int32_t>(train > 0 ? train : 4096,
                                                    4096));
    cp.n_batch = options.batch_tokens ? options.batch_tokens : 512;
    cp.n_ubatch = std::min(cp.n_batch, cp.n_ubatch ? cp.n_ubatch : 512u);
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = flash;
    cp.no_perf = true;
    ctx.lctx = llama_init_from_model(ctx.model, cp);
    if (!ctx.lctx)
        throw std::runtime_error("failed to create llama context (n_ctx=" +
                                 std::to_string(cp.n_ctx) + ")");
    llama_set_abort_callback(ctx.lctx, abort_callback, &ctx);

    if (options.mmproj_path && *options.mmproj_path) {
        mtmd_context_params mtp = mtmd_context_params_default();
        mtp.use_gpu = dev != nullptr;
        mtp.device = dev;
        mtp.print_timings = false;
        mtp.n_threads = threads;
        mtp.flash_attn_type = flash;
        if (options.image_max_tokens > 0)
            mtp.image_max_tokens = options.image_max_tokens;
        ctx.mctx = mtmd_init_from_file(options.mmproj_path, ctx.model, mtp);
        if (!ctx.mctx)
            throw std::runtime_error(
                    std::string("projector failed to load or does not match "
                                "the text model: ") +
                    options.mmproj_path);
        if (!mtmd_support_vision(ctx.mctx))
            throw std::runtime_error("projector has no vision encoder");
    }
}

std::string apply_template(const llama_model* model,
                           const std::vector<llama_chat_message>& messages) {
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) throw std::runtime_error("model has no chat template");
    std::vector<char> buffer(4096);
    int32_t n = llama_chat_apply_template(tmpl, messages.data(),
                                          messages.size(), true, buffer.data(),
                                          int32_t(buffer.size()));
    if (n > int32_t(buffer.size())) {
        buffer.resize(size_t(n));
        n = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                      true, buffer.data(),
                                      int32_t(buffer.size()));
    }
    if (n < 0)
        throw std::runtime_error(
                "chat template is not supported by llama_chat_apply_template");
    return std::string(buffer.data(), size_t(n));
}

int generate(aicore_llm_ctx& ctx,
             const aicore_llm_request& request,
             OwnedResult& out) {
    const auto start = Clock::now();
    if (!request.messages || request.message_count == 0)
        throw std::runtime_error("request has no messages");
    if (request.image_count && !request.images)
        throw std::runtime_error("image_count set without images");
    if (request.image_count && !ctx.mctx)
        throw std::runtime_error("context has no vision projector");
    if (request.stop_count && !request.stop_sequences)
        throw std::runtime_error("stop_count set without stop_sequences");
    if (!std::isfinite(request.temperature) || request.temperature < 0 ||
        (request.temperature > 0 &&
         (!std::isfinite(request.top_p) || request.top_p <= 0 ||
          request.top_p > 1)))
        throw std::runtime_error("invalid temperature/top_p");
    const uint32_t n_ctx = llama_n_ctx(ctx.lctx);
    if (request.max_tokens == 0 || request.max_tokens > n_ctx)
        throw std::runtime_error("max_tokens must be 1..context size");

    // ---- preprocess: messages -> rendered prompt -> chunks/tokens
    const std::string marker = mtmd_default_marker();
    std::vector<std::string> contents(request.message_count);
    size_t markers = 0;
    int last_user = -1;
    for (size_t i = 0; i < request.message_count; ++i) {
        const auto& m = request.messages[i];
        if (!m.role || !m.content)
            throw std::runtime_error("message role/content must be set");
        contents[i] = m.content;
        markers += count_occurrences(contents[i], marker);
        if (std::strcmp(m.role, "user") == 0) last_user = int(i);
    }
    if (markers == 0 && request.image_count) {
        if (last_user < 0)
            throw std::runtime_error("images require a user message");
        std::string prefix;
        for (size_t i = 0; i < request.image_count; ++i) prefix += marker;
        contents[size_t(last_user)] = prefix + contents[size_t(last_user)];
        markers = request.image_count;
    }
    if (markers != request.image_count)
        throw std::runtime_error("media marker count (" +
                                 std::to_string(markers) +
                                 ") != image_count (" +
                                 std::to_string(request.image_count) + ")");
    std::vector<llama_chat_message> chat(request.message_count);
    for (size_t i = 0; i < request.message_count; ++i)
        chat[i] = {request.messages[i].role, contents[i].c_str()};
    const std::string prompt = apply_template(ctx.model, chat);

    const llama_vocab* vocab = llama_model_get_vocab(ctx.model);
    llama_memory_clear(llama_get_memory(ctx.lctx), true);
    std::vector<llama_token> text_tokens;
    std::unique_ptr<mtmd_input_chunks, ChunksDeleter> chunks;
    if (ctx.mctx) {
        std::vector<std::unique_ptr<mtmd_bitmap, BitmapDeleter>> bitmaps;
        std::vector<const mtmd_bitmap*> bitmap_ptrs;
        std::vector<uint8_t> rgb;
        for (size_t i = 0; i < request.image_count; ++i) {
            aicore::img::Image view;
            if (!aicore::img::image_from_view(request.images[i], view))
                throw std::runtime_error("invalid image view/stride/format");
            rgb.resize(size_t(view.width) * size_t(view.height) * 3);
            for (int y = 0; y < view.height; ++y) {
                const uint8_t* row = view.row(y);
                uint8_t* dst = rgb.data() + size_t(y) * size_t(view.width) * 3;
                for (int x = 0; x < view.width; ++x)
                    for (int c = 0; c < 3; ++c)
                        dst[x * 3 + c] =
                                row[x * view.channels + view.channel_index(c)];
            }
            bitmaps.emplace_back(mtmd_bitmap_init(uint32_t(view.width),
                                                  uint32_t(view.height),
                                                  rgb.data()));
            if (!bitmaps.back())
                throw std::runtime_error("failed to create image bitmap");
            bitmap_ptrs.push_back(bitmaps.back().get());
        }
        chunks.reset(mtmd_input_chunks_init());
        mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
        const int32_t rc = mtmd_tokenize(ctx.mctx, chunks.get(), &text,
                                         bitmap_ptrs.data(),
                                         bitmap_ptrs.size());
        if (rc != 0)
            throw std::runtime_error("mtmd_tokenize failed (" +
                                     std::to_string(rc) + ")");
        out.public_result.prompt_tokens =
                uint32_t(mtmd_helper_get_n_tokens(chunks.get()));
    } else {
        int32_t n = -llama_tokenize(vocab, prompt.c_str(),
                                    int32_t(prompt.size()), nullptr, 0, true,
                                    true);
        text_tokens.resize(size_t(std::max(n, 0)));
        if (n <= 0 ||
            llama_tokenize(vocab, prompt.c_str(), int32_t(prompt.size()),
                           text_tokens.data(), n, true, true) != n)
            throw std::runtime_error("prompt tokenization failed");
        out.public_result.prompt_tokens = uint32_t(text_tokens.size());
    }
    const auto preprocessed = Clock::now();

    // ---- inference: prefill
    llama_pos n_past = 0;
    const int32_t n_batch = int32_t(llama_n_batch(ctx.lctx));
    bool cancelled = false;
    if (ctx.mctx) {
        if (mtmd_helper_get_n_pos(chunks.get()) >= llama_pos(n_ctx))
            throw std::runtime_error("prompt exceeds the context size");
        const int32_t rc = mtmd_helper_eval_chunks(
                ctx.mctx, ctx.lctx, chunks.get(), 0, 0, n_batch, true, &n_past);
        if (rc != 0) {
            if (!ctx.cancelled())
                throw std::runtime_error("prefill failed (" +
                                         std::to_string(rc) + ")");
            cancelled = true;
        }
    } else {
        if (text_tokens.size() >= n_ctx)
            throw std::runtime_error("prompt exceeds the context size");
        for (size_t at = 0; at < text_tokens.size() && !cancelled;
             at += size_t(n_batch)) {
            const int32_t n = int32_t(
                    std::min(text_tokens.size() - at, size_t(n_batch)));
            const int32_t rc = llama_decode(
                    ctx.lctx, llama_batch_get_one(text_tokens.data() + at, n));
            if (rc != 0) {
                if (!ctx.cancelled())
                    throw std::runtime_error("prefill failed (" +
                                             std::to_string(rc) + ")");
                cancelled = true;
            }
        }
        n_past = llama_pos(text_tokens.size());
    }
    const auto prefilled = Clock::now();

    // ---- inference: decode loop
    std::unique_ptr<llama_sampler, SamplerDeleter> sampler(
            llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (request.temperature > 0) {
        llama_sampler_chain_add(sampler.get(),
                                llama_sampler_init_top_p(request.top_p, 1));
        llama_sampler_chain_add(sampler.get(),
                                llama_sampler_init_temp(request.temperature));
        llama_sampler_chain_add(sampler.get(),
                                llama_sampler_init_dist(request.seed));
    } else {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    }
    std::vector<std::string> stops;
    size_t holdback = 0;
    for (size_t i = 0; i < request.stop_count; ++i) {
        if (!request.stop_sequences[i] || !*request.stop_sequences[i]) continue;
        stops.emplace_back(request.stop_sequences[i]);
        holdback = std::max(holdback, stops.back().size() - 1);
    }
    uint32_t finish = AICORE_LLM_FINISH_LENGTH;
    size_t streamed = 0;
    auto stream_upto = [&](size_t end) {
        end = complete_utf8_prefix(out.text, end);
        if (end <= streamed) return;
        if (request.stream_cb &&
            request.stream_cb(out.text.data() + streamed, end - streamed,
                              request.stream_user_data) != 0)
            cancelled = true;
        streamed = end;
    };
    char piece[256];
    while (!cancelled && out.tokens.size() < request.max_tokens) {
        if (ctx.cancelled()) {
            cancelled = true;
            break;
        }
        if (n_past >= llama_pos(n_ctx)) break;  // context exhausted: LENGTH
        llama_token token = llama_sampler_sample(sampler.get(), ctx.lctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            finish = AICORE_LLM_FINISH_STOP;
            break;
        }
        out.tokens.push_back(token);
        const int32_t n = llama_token_to_piece(vocab, token, piece,
                                               int32_t(sizeof(piece)), 0,
                                               false);
        if (n < 0) throw std::runtime_error("token detokenization failed");
        const size_t before = out.text.size();
        out.text.append(piece, size_t(n));
        bool stopped = false;
        for (const auto& stop : stops) {
            // Only the window that can contain a new match is searched.
            const size_t from =
                    before >= stop.size() ? before - stop.size() + 1 : 0;
            const size_t at = out.text.find(stop, from);
            if (at != std::string::npos) {
                out.text.resize(at);
                stopped = true;
            }
        }
        if (stopped) {
            finish = AICORE_LLM_FINISH_STOP;
            break;
        }
        stream_upto(out.text.size() > holdback ? out.text.size() - holdback
                                               : 0);
        if (cancelled) break;
        const int32_t rc =
                llama_decode(ctx.lctx, llama_batch_get_one(&token, 1));
        if (rc != 0) {
            if (!ctx.cancelled())
                throw std::runtime_error("decode failed (" +
                                         std::to_string(rc) + ")");
            cancelled = true;
            break;
        }
        ++n_past;
    }
    if (cancelled) finish = AICORE_LLM_FINISH_CANCELLED;
    else
        stream_upto(out.text.size());
    const auto decoded = Clock::now();

    // ---- postprocess: typed result
    auto& r = out.public_result;
    r.struct_size = sizeof(aicore_llm_result);
    r.abi_version = AICORE_LLM_ABI_VERSION;
    r.text = out.text.c_str();
    r.text_bytes = out.text.size();
    r.finish_reason = finish;
    r.generated_tokens = uint32_t(out.tokens.size());
    r.token_ids = out.tokens.empty() ? nullptr : out.tokens.data();
    r.prefill_ms = ms_since(preprocessed, prefilled);
    r.decode_ms = ms_since(prefilled, decoded);
    r.private_owner = &out;
    const auto end = Clock::now();
    auto& t = r.timings;
    t.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    t.valid_fields = AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
                     AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    t.preprocess_ms = ms_since(start, preprocessed);
    t.inference_ms = ms_since(preprocessed, decoded);
    t.postprocess_ms = ms_since(decoded, end);
    t.e2e_ms = ms_since(start, end);
    return cancelled ? 1 : 0;
}

}  // namespace

aicore_llm_options aicore_llm_default_options(void) {
    aicore_llm_options result{};
    result.struct_size = sizeof(result);
    result.abi_version = AICORE_LLM_ABI_VERSION;
    result.device = "auto";
    result.flash_attn = -1;
    return result;
}
aicore_llm_request aicore_llm_default_request(void) {
    aicore_llm_request result{};
    result.struct_size = sizeof(result);
    result.abi_version = AICORE_LLM_ABI_VERSION;
    result.max_tokens = 512;
    result.top_p = 0.95f;
    return result;
}
aicore_llm_ctx* aicore_llm_create(const aicore_llm_options* options) {
    std::unique_ptr<aicore_llm_ctx> ctx(new (std::nothrow) aicore_llm_ctx);
    if (!ctx) return nullptr;
    try {
        if (!valid(options))
            throw std::runtime_error("invalid options ABI/size");
        load(*ctx, *options);
    } catch (const std::exception& e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unknown llm load error";
    }
    if (!ctx->error.empty()) {
        // An error context owns no runtime objects.
        if (ctx->mctx) mtmd_free(ctx->mctx);
        if (ctx->lctx) llama_free(ctx->lctx);
        if (ctx->model) llama_model_free(ctx->model);
        ctx->mctx = nullptr;
        ctx->lctx = nullptr;
        ctx->model = nullptr;
    }
    return ctx.release();
}
void aicore_llm_free(aicore_llm_ctx* ctx) { delete ctx; }
int aicore_llm_ready(const aicore_llm_ctx* ctx) {
    return ctx && ctx->ready() ? 1 : 0;
}
const char* aicore_llm_last_error(const aicore_llm_ctx* ctx) {
    return ctx ? ctx->error.c_str() : "null context";
}
const char* aicore_llm_context_device(const aicore_llm_ctx* ctx) {
    return aicore_llm_ready(ctx) ? ctx->device.c_str() : "";
}
int aicore_llm_supports_vision(const aicore_llm_ctx* ctx) {
    return aicore_llm_ready(ctx) && ctx->mctx ? 1 : 0;
}
const char* aicore_llm_media_marker(void) { return mtmd_default_marker(); }
int aicore_llm_generate(aicore_llm_ctx* ctx,
                        const aicore_llm_request* request,
                        aicore_llm_result** output) {
    if (output) *output = nullptr;
    if (!ctx) return -1;
    ctx->has_timings = false;
    ctx->timings = {};
    ctx->cancel.store(false, std::memory_order_relaxed);
    try {
        if (!output || !valid(request) || !ctx->ready())
            throw std::runtime_error("invalid request, output or context");
        auto result = std::make_unique<OwnedResult>();
        const int rc = generate(*ctx, *request, *result);
        ctx->timings = result->public_result.timings;
        ctx->has_timings = true;
        ctx->error.clear();
        *output = &result.release()->public_result;
        return rc;
    } catch (const std::exception& e) {
        ctx->error = e.what();
    } catch (...) {
        ctx->error = "unknown llm generation error";
    }
    return -1;
}
void aicore_llm_cancel(aicore_llm_ctx* ctx) {
    if (ctx) ctx->cancel.store(true, std::memory_order_relaxed);
}
void aicore_llm_result_free(aicore_llm_result* result) {
    if (result) delete static_cast<OwnedResult*>(result->private_owner);
}
int aicore_llm_last_pipeline_timings(const aicore_llm_ctx* ctx,
                                     aicore_pipeline_timings* output) {
    if (!ctx || !output || !ctx->has_timings) return -1;
    *output = ctx->timings;
    return 0;
}
char* aicore_llm_model_cache_dir(void) {
    try {
        const auto path = aicore::extract_model_dir("llm_models");
        auto* result = static_cast<char*>(std::malloc(path.size() + 1));
        if (result) std::memcpy(result, path.c_str(), path.size() + 1);
        return result;
    } catch (...) {
        return nullptr;
    }
}
void aicore_llm_free_buffer(void* buffer) { std::free(buffer); }
void aicore_llm_shutdown(void) { aicore_runtime_shutdown(); }
