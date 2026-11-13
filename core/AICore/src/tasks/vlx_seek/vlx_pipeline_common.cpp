// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Shared pipeline helpers — bodies moved verbatim from vlx_cli.cpp (see the
// header note). Behavior is frozen by the benchmark regression suite.
#include "tasks/vlx_seek/vlx_pipeline_common.h"

#include <FileSystem.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>

#include "tasks/vlx_seek/vlx_image_prep.h"

namespace vlx {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::string py_float(double v) {
    char buf[64];
    const double a = std::fabs(v);
    if (a == 0.0 || (a >= 1e-4 && a < 1e16)) {
        // snprintf with %.17g-equivalent fixed formatting. std::to_chars for
        // float requires macOS 13.3+ / glibc 2.29+, but the deployment target
        // is macOS 11.0, so we use the universally available snprintf.
        int n = std::snprintf(buf, sizeof(buf), "%.15f", v);
        std::string s(buf, n);
        // Trim trailing zeros after the decimal point, keeping at least one.
        if (s.find('.') != std::string::npos) {
            while (s.size() > 1 && s.back() == '0' && s[s.size() - 2] != '.')
                s.pop_back();
        }
        if (s.find('.') == std::string::npos) s += ".0";
        return s;
    }
    int n = std::snprintf(buf, sizeof(buf), "%.15e", v);
    return std::string(buf, n);
}

std::string py_json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                o += "\\\"";
                break;
            case '\\':
                o += "\\\\";
                break;
            case '\n':
                o += "\\n";
                break;
            case '\r':
                o += "\\r";
                break;
            case '\t':
                o += "\\t";
                break;
            case '\b':
                o += "\\b";
                break;
            case '\f':
                o += "\\f";
                break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += char(c);
                }
        }
    }
    return o + "\"";
}

std::string boxes_to_json(const std::vector<Box>& boxes) {
    std::string s = "[";
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& b = boxes[i];
        s += (i ? ", [" : "[") + py_float(b.x1) + ", " + py_float(b.y1) + ", " +
             py_float(b.x2) + ", " + py_float(b.y2) + "]";
    }
    return s + "]";
}

// Debug dump sink for the option-gated parity dumps (vlx_pipeline_common.h).
// Returns false silently when the file cannot be opened: dump failures must
// never break the run that produced the tensors.
bool write_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream file = cloudViewer::utility::filesystem::OpenOutputStream(
            path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    if (bytes && data) {
        file.write(static_cast<const char*>(data),
                   static_cast<std::streamsize>(bytes));
    }
    return file.good();
}

bool validate_boxes(const std::vector<std::vector<double>>& raw,
                    int w,
                    int h,
                    std::vector<BoxD>* out,
                    std::string* err) {
    out->clear();
    for (const auto& b : raw) {
        if (b.size() != 4) {
            *err = "Each bbox must contain [x1, y1, x2, y2].";
            return false;
        }
        auto clip = [](double v, double hi) {
            return std::max(0.0, std::min(v, hi));
        };
        double x1 = clip(b[0], w), x2 = clip(b[2], w), y1 = clip(b[1], h),
               y2 = clip(b[3], h);
        if (x2 < x1) std::swap(x1, x2);
        if (y2 < y1) std::swap(y1, y2);
        if (!(x2 > x1) || !(y2 > y1)) {
            *err = "bbox must have positive area after clipping";
            return false;
        }
        out->push_back({x1, y1, x2, y2});
    }
    return true;
}

std::vector<int> raster_order(const std::vector<BoxD>& boxes) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (boxes[a][0] != boxes[b][0]) return boxes[a][0] < boxes[b][0];
        if (boxes[a][1] != boxes[b][1]) return boxes[a][1] < boxes[b][1];
        return a < b;
    });
    return order;
}

std::string remap_obj_tokens(const std::string& s,
                             const std::map<long long, long long>& m) {
    std::string o;
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, 4, "<obj") == 0) {
            size_t j = i + 4;
            while (j < s.size() &&
                   std::isdigit(static_cast<unsigned char>(s[j])))
                ++j;
            if (j > i + 4 && j < s.size() && s[j] == '>' && j - (i + 4) <= 18) {
                const long long idx = std::stoll(s.substr(i + 4, j - i - 4));
                const auto it = m.find(idx);
                o += it == m.end() ? s.substr(i, j + 1 - i)
                                   : "<obj" + std::to_string(it->second) + ">";
                i = j + 1;
                continue;
            }
        }
        o += s[i++];
    }
    return o;
}

bool build_task_prompt(const std::string& task,
                       const std::string& lang,
                       const std::string& text,
                       std::string* out) {
    static const struct {
        const char* lang;
        const char* task;
        const char* tpl;
    } kTemplates[] = {
#include "vlx_task_templates.inc"
    };
    for (const auto& t : kTemplates) {
        if (task != t.task || lang != t.lang) continue;
        const std::string tpl = t.tpl;
        const size_t p = tpl.find("{}");
        *out = tpl.substr(0, p) + text +
               tpl.substr(p + 2);  // str.format, one slot
        return true;
    }
    return false;
}

// Qwen2VLImageProcessorFast (preprocessor_config.json): smart_resize(factor
// 32, min 65536, max 16777216 px) -> torch uint8 AA bicubic -> (x - 127.5) /
// 127.5 in f32 -> patchify (t duplicated) -> pixel_values; the tower casts to
// bf16. Returns token-major (N, 1536) BEFORE the bf16 cast (= the dumped
// pixel_values) and the bf16 copy fed to the graph.
void primary_pixels(const ImageU8& img,
                    int* gh,
                    int* gw,
                    std::vector<float>* f32,
                    std::vector<float>* bf16) {
    int hb = 0, wb = 0;
    smart_resize(img.height, img.width, 32, 65536.0, 16777216.0, &hb, &wb);
    const ImageU8 rs = resize_u8(img, wb, hb, ResampleFilter::Bicubic,
                                 ResampleImpl::kTorch);
    const int patch = 16, merge = 2, temp = 2;
    *gh = hb / patch;
    *gw = wb / patch;
    const int K = 3 * temp * patch * patch;
    f32->assign(size_t(*gh) * *gw * K, 0.f);
    for (int bh = 0; bh < *gh / merge; ++bh)
        for (int bw = 0; bw < *gw / merge; ++bw)
            for (int mh = 0; mh < merge; ++mh)
                for (int mw = 0; mw < merge; ++mw) {
                    const int p =
                            ((bh * (*gw / merge) + bw) * merge + mh) * merge +
                            mw;
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < temp; ++t)
                            for (int iy = 0; iy < patch; ++iy)
                                for (int ix = 0; ix < patch; ++ix) {
                                    const int y =
                                            (bh * merge + mh) * patch + iy;
                                    const int x =
                                            (bw * merge + mw) * patch + ix;
                                    const size_t k = size_t(c * temp + t) *
                                                             patch * patch +
                                                     iy * patch + ix;
                                    const float u = float(
                                            rs.rgb[(size_t(y) * wb + x) * 3 +
                                                   c]);
                                    (*f32)[size_t(p) * K + k] =
                                            (u - 127.5f) / 127.5f;
                                }
                }
    bf16->resize(f32->size());
    for (size_t i = 0; i < f32->size(); ++i) (*bf16)[i] = bf16_round((*f32)[i]);
}

// Aux tower input: CLIPImageProcessor(do_resize=False, rescale only:
// f32(u8) * f32(1/255)) -> .to(bf16) -> _resize_to_supported (nearest
// multiple of 16 per side, banker's rounding; AA bilinear on f32) -> bf16.
// Returns CHW (3, H, W) bf16 values and the HWC copy RadioGraph consumes.
void aux_pixels(const ImageU8& img,
                int* H,
                int* W,
                std::vector<float>* chw,
                std::vector<float>* hwc) {
    const int h0 = img.height, w0 = img.width;
    const float scale = float(1.0 / 255.0);
    std::vector<float> src(size_t(3) * h0 * w0);
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < size_t(h0) * w0; ++i)
            src[size_t(c) * h0 * w0 + i] =
                    bf16_round(float(img.rgb[i * 3 + c]) * scale);
    auto nearest16 = [](int v) {
        return std::max(int(std::nearbyint(v / 16.0)) * 16, 16);
    };
    *H = nearest16(h0);
    *W = nearest16(w0);
    if (*H == h0 && *W == w0) {
        *chw = std::move(src);
    } else {
        *chw = resize_f32_bilinear_aa(src, 3, h0, w0, *H, *W);
        for (float& v : *chw) v = bf16_round(v);
    }
    const size_t n = size_t(*H) * *W;
    hwc->resize(n * 3);
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < n; ++i)
            (*hwc)[i * 3 + c] = (*chw)[size_t(c) * n + i];
}

// ---- llama.cpp bridge ----

// Tokenize without BOS, special tokens parsed (HF tokenizer(chunk) on Qwen).
std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text) {
    // Retry with a growing buffer instead of exiting: special-token parsing
    // can expand the byte count, so the first estimate may overflow.
    std::vector<llama_token> toks(text.size() + 8);
    int n = llama_tokenize(vocab, text.data(), int(text.size()), toks.data(),
                           int(toks.size()), false, true);
    for (int grow = 0; n < 0 && grow < 8; ++grow) {
        toks.resize(toks.size() * 2);
        n = llama_tokenize(vocab, text.data(), int(text.size()), toks.data(),
                           int(toks.size()), false, true);
    }
    toks.resize(n < 0 ? 0 : size_t(n));
    return toks;
}

std::string token_piece(const llama_vocab* vocab, llama_token t) {
    char buf[256];
    const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, size_t(n)) : std::string();
}

int32_t decode_entries(llama_context* ctx,
                       const std::vector<Entry>& entries,
                       int64_t n_embd,
                       int64_t* pos_cursor) {
    llama_batch_ext* b = llama_batch_ext_init(ctx);
    for (size_t i = 0; i < entries.size(); ++i) {
        const llama_pos p = llama_pos(*pos_cursor + int64_t(i));
        const llama_pos pos4[4] = {p, p, p, 0};
        int32_t idx;
        if (entries[i].id != LLAMA_TOKEN_NULL) {
            idx = llama_batch_ext_add_token(b, 0, entries[i].id);
        } else {
            // n_rows = 1: one batch entry is one position
            idx = llama_batch_ext_add_embd(
                    b, 0, llama_embd{entries[i].embd, 1, size_t(n_embd)});
        }
        if (idx < 0) {
            llama_batch_ext_free(b);
            return idx;
        }
        llama_batch_ext_set_pos(b, idx, pos4);
        if (i + 1 == entries.size())
            llama_batch_ext_set_output_logits(b, idx, true);
    }
    const int32_t r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
    *pos_cursor += int64_t(entries.size());
    llama_batch_ext_free(b);
    return r;
}

llama_token select_token(const float* logits,
                         int n_vocab,
                         float temperature,
                         float top_p,
                         std::mt19937* rng) {
    int best = 0;
    for (int v = 1; v < n_vocab; ++v)
        if (logits[v] > logits[best]) best = v;
    if (temperature <= 0.f) return best;
    std::vector<std::pair<float, int>> p(static_cast<size_t>(n_vocab));
    double z = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        p[v] = {std::exp((logits[v] - logits[best]) / temperature), v};
        z += p[v].first;
    }
    std::sort(p.begin(), p.end(),
              [](auto& a, auto& b) { return a.first > b.first; });
    size_t keep = p.size();
    if (top_p < 1.f) {  // TopPLogitsWarper: smallest set with mass >= top_p
        double cum = 0.0;
        for (size_t i = 0; i < p.size(); ++i) {
            cum += p[i].first / z;
            if (cum >= top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    std::vector<double> w(keep);
    for (size_t i = 0; i < keep; ++i) w[i] = p[i].first;
    std::discrete_distribution<size_t> dist(w.begin(), w.end());
    return p[dist(*rng)].second;
}

std::vector<llama_token> generate(llama_context* ctx,
                                  const llama_vocab* vocab,
                                  int max_new,
                                  const std::vector<llama_token>& stop_ids,
                                  float temperature,
                                  float top_p,
                                  int64_t* pos_cursor,
                                  double* decode_ms) {
    std::vector<llama_token> out;
    std::mt19937 rng(0);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const auto t0 = Clock::now();
    for (int i = 0; i < max_new; ++i) {
        const llama_token tok = select_token(llama_get_logits_ith(ctx, -1),
                                             n_vocab, temperature, top_p, &rng);
        out.push_back(tok);
        if (std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end())
            break;
        if (i + 1 == max_new) break;
        if (decode_entries(ctx, {{tok, nullptr}}, 0, pos_cursor)) {
            break;  // decode failed: the caller surfaces the non-zero code
        }
    }
    *decode_ms = ms_since(t0);
    return out;
}

// ---- official-style drawing ----

void draw_rect_pil(std::vector<uint8_t>* rgb,
                   int w,
                   int h,
                   double fx0,
                   double fy0,
                   double fx1,
                   double fy1,
                   int width) {
    int x0 = int(fx0), y0 = int(fy0), x1 = int(fx1), y1 = int(fy1);
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);
    auto put = [&](int x, int y) {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        uint8_t* p = rgb->data() + (size_t(y) * w + x) * 3;
        p[0] = 255;
        p[1] = 0;
        p[2] = 0;
    };
    for (int i = 0; i < width; ++i) {
        for (int x = x0; x <= x1; ++x) {
            put(x, y0 + i);
            put(x, y1 - i);
        }
        for (int y = y0 + width; y <= y1 - width; ++y) {
            put(x0 + i, y);
            put(x1 - i, y);
        }
    }
}

const uint8_t kFont5x7[95][5] = {
        {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x5F, 0x00, 0x00},
        {0x07, 0x07, 0x00, 0x00, 0x00}, {0x14, 0x7F, 0x14, 0x7F, 0x14},
        {0x24, 0x2A, 0x7F, 0x2A, 0x12}, {0x23, 0x13, 0x08, 0x64, 0x62},
        {0x36, 0x49, 0x55, 0x22, 0x50}, {0x05, 0x03, 0x00, 0x00, 0x00},
        {0x00, 0x1C, 0x22, 0x41, 0x00}, {0x00, 0x41, 0x22, 0x1C, 0x00},
        {0x14, 0x08, 0x3E, 0x08, 0x14}, {0x08, 0x08, 0x3E, 0x08, 0x08},
        {0x00, 0x50, 0x30, 0x00, 0x00}, {0x08, 0x08, 0x08, 0x08, 0x08},
        {0x00, 0x60, 0x60, 0x00, 0x00}, {0x20, 0x10, 0x08, 0x04, 0x02},
        {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00},
        {0x42, 0x61, 0x51, 0x49, 0x46}, {0x21, 0x41, 0x45, 0x4B, 0x31},
        {0x18, 0x14, 0x12, 0x7F, 0x10}, {0x27, 0x45, 0x45, 0x45, 0x39},
        {0x3C, 0x4A, 0x49, 0x49, 0x30}, {0x01, 0x71, 0x09, 0x05, 0x03},
        {0x36, 0x49, 0x49, 0x49, 0x36}, {0x06, 0x49, 0x49, 0x29, 0x1E},
        {0x00, 0x36, 0x36, 0x00, 0x00}, {0x00, 0x56, 0x36, 0x00, 0x00},
        {0x08, 0x14, 0x22, 0x41, 0x00}, {0x14, 0x14, 0x14, 0x14, 0x14},
        {0x00, 0x41, 0x22, 0x14, 0x08}, {0x02, 0x01, 0x51, 0x09, 0x06},
        {0x32, 0x49, 0x79, 0x41, 0x3E}, {0x7E, 0x11, 0x11, 0x11, 0x7E},
        {0x7F, 0x49, 0x49, 0x49, 0x36}, {0x3E, 0x41, 0x41, 0x41, 0x22},
        {0x7F, 0x41, 0x41, 0x22, 0x1C}, {0x7F, 0x49, 0x49, 0x49, 0x41},
        {0x7F, 0x09, 0x09, 0x09, 0x01}, {0x3E, 0x41, 0x49, 0x49, 0x7A},
        {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00},
        {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41},
        {0x7F, 0x40, 0x40, 0x40, 0x40}, {0x7F, 0x02, 0x0C, 0x02, 0x7F},
        {0x7F, 0x04, 0x08, 0x10, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E},
        {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E},
        {0x7F, 0x09, 0x19, 0x29, 0x46}, {0x46, 0x49, 0x49, 0x49, 0x31},
        {0x01, 0x01, 0x7F, 0x01, 0x01}, {0x3F, 0x40, 0x40, 0x40, 0x3F},
        {0x1F, 0x20, 0x40, 0x20, 0x1F}, {0x3F, 0x40, 0x38, 0x40, 0x3F},
        {0x63, 0x14, 0x08, 0x14, 0x63}, {0x07, 0x08, 0x70, 0x08, 0x07},
        {0x61, 0x51, 0x49, 0x45, 0x43}, {0x00, 0x7F, 0x41, 0x41, 0x00},
        {0x02, 0x04, 0x08, 0x10, 0x20}, {0x00, 0x41, 0x41, 0x7F, 0x00},
        {0x04, 0x02, 0x01, 0x02, 0x04}, {0x40, 0x40, 0x40, 0x40, 0x40},
        {0x00, 0x01, 0x02, 0x04, 0x00}, {0x20, 0x54, 0x54, 0x54, 0x78},
        {0x7F, 0x48, 0x44, 0x44, 0x38}, {0x38, 0x44, 0x44, 0x44, 0x20},
        {0x38, 0x44, 0x44, 0x48, 0x7F}, {0x38, 0x54, 0x54, 0x54, 0x18},
        {0x08, 0x7E, 0x09, 0x01, 0x02}, {0x0C, 0x52, 0x52, 0x52, 0x3E},
        {0x7F, 0x08, 0x04, 0x04, 0x78}, {0x00, 0x44, 0x7D, 0x40, 0x00},
        {0x20, 0x40, 0x44, 0x3D, 0x00}, {0x7F, 0x10, 0x28, 0x44, 0x00},
        {0x00, 0x41, 0x7F, 0x40, 0x00}, {0x7C, 0x04, 0x18, 0x04, 0x78},
        {0x7C, 0x08, 0x04, 0x04, 0x78}, {0x38, 0x44, 0x44, 0x44, 0x38},
        {0x7C, 0x14, 0x14, 0x14, 0x08}, {0x08, 0x14, 0x14, 0x18, 0x7C},
        {0x7C, 0x08, 0x04, 0x04, 0x08}, {0x48, 0x54, 0x54, 0x54, 0x20},
        {0x04, 0x3F, 0x44, 0x40, 0x20}, {0x3C, 0x40, 0x40, 0x20, 0x7C},
        {0x1C, 0x20, 0x40, 0x20, 0x1C}, {0x3C, 0x40, 0x30, 0x40, 0x3C},
        {0x44, 0x28, 0x10, 0x28, 0x44}, {0x0C, 0x50, 0x50, 0x50, 0x3C},
        {0x44, 0x64, 0x54, 0x4C, 0x44}, {0x00, 0x08, 0x36, 0x41, 0x00},
        {0x00, 0x00, 0x7F, 0x00, 0x00}, {0x00, 0x41, 0x36, 0x08, 0x00},
        {0x08, 0x08, 0x2A, 0x1C, 0x08},
};

void draw_label_pil(std::vector<uint8_t>* rgb,
                    int w,
                    int h,
                    double fx,
                    double fy,
                    const std::string& label) {
    const int scale = w > 640 ? 2 : 1;
    int x = int(fx) + 2, y = int(fy) + 2;
    for (unsigned char ch : label) {
        if (ch < 0x20 || ch > 0x7E) ch = '?';
        const auto* col = kFont5x7[ch - 0x20];
        for (int cx = 0; cx < 5; ++cx)
            for (int cy = 0; cy < 7; ++cy)
                if (col[cx] & (1u << cy))
                    for (int sy = 0; sy < scale; ++sy)
                        for (int sx = 0; sx < scale; ++sx) {
                            const int px = x + cx * scale + sx,
                                      py = y + cy * scale + sy;
                            if (px < 0 || px >= w || py < 0 || py >= h)
                                continue;
                            uint8_t* p =
                                    rgb->data() + (size_t(py) * w + px) * 3;
                            p[0] = 255;
                            p[1] = 0;
                            p[2] = 0;
                        }
        x += 6 * scale;
    }
}

bool all_ascii(const std::string& s) {
    for (unsigned char c : s)
        if (c > 0x7E) return false;
    return true;
}

}  // namespace vlx
