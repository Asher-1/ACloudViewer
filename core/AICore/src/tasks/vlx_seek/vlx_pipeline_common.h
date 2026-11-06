// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Shared pipeline helpers for the VLX-Seek task (AICore port of the upstream
// Asher-1/VLX-Seek-GGML cpp_ggml/src/vlx_pipeline_common.*).
//
// Single home for the pieces the task runtime needs: llama.cpp
// tokenization/decode bridge, official-format JSON serialization, task
// templates, host preprocessing and the official-style box/label drawing.
// Numeric behavior is frozen by the upstream benchmark regression suite.
#pragma once

#include <array>
#include <chrono>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "llama.h"
#include "tasks/vlx_seek/vlx_types.h"
#include "vlx_image_prep.h"  // ImageU8

namespace vlx {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0);

std::string lower_ascii(std::string s);

// Debug dump sink for the option-gated parity dumps (empty path = no-op).
bool write_file(const std::string& path, const void* data, size_t bytes);

// float.__repr__: shortest round-trip digits, fixed notation in
// [1e-4, 1e16) with a mandatory ".0", else 'e' notation ("1e-05").
std::string py_float(double v);
// json.dumps(..., ensure_ascii=False) string escaping.
std::string py_json_str(const std::string& s);
// Official-format boxes.json: [[x1, y1, x2, y2], ...] as
// json.dump(tensor.tolist()).
std::string boxes_to_json(const std::vector<Box>& boxes);

using BoxD = std::array<double, 4>;  // python float xyxy (caller-facing)

// _validate_boxes: clip to the image, sort each axis, reject empty boxes.
bool validate_boxes(const std::vector<std::vector<double>>& raw,
                    int w,
                    int h,
                    std::vector<BoxD>* out,
                    std::string* err);

// _order_boxes, mm_bbox_order_mode = "raster": key (x1, y1, caller index).
std::vector<int> raster_order(const std::vector<BoxD>& boxes);

// re.sub(r"<obj(\d+)>", ...) with an index map; unmapped indices are kept.
std::string remap_obj_tokens(const std::string& s,
                             const std::map<long long, long long>& m);

bool build_task_prompt(const std::string& task,
                       const std::string& lang,
                       const std::string& text,
                       std::string* out);

// Qwen2VLImageProcessorFast primary-tower pixels (token-major f32 + bf16 copy)
// and CLIPImageProcessor aux-tower pixels (CHW/hwc bf16).
void primary_pixels(const ImageU8& img,
                    int* gh,
                    int* gw,
                    std::vector<float>* f32,
                    std::vector<float>* bf16);
void aux_pixels(const ImageU8& img,
                int* H,
                int* W,
                std::vector<float>* chw,
                std::vector<float>* hwc);

// ---- llama.cpp bridge ----

// One prompt entry: a vocab token, or an injected embedding row.
struct Entry {
    llama_token id;     // LLAMA_TOKEN_NULL for embedding rows
    const float* embd;  // n_embd floats when id == LLAMA_TOKEN_NULL
};

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text);
std::string token_piece(const llama_vocab* vocab, llama_token t);
// Single llama_process over the whole sequence: tokens get position p,
// embedding rows get the 4 M-RoPE sections {p, p, p, 0}. Logits are
// requested for the last entry only.
int32_t decode_entries(llama_context* ctx,
                       const std::vector<Entry>& entries,
                       int64_t n_embd,
                       int64_t* pos_cursor);
// HF generate token selection: argmax (first max on ties, = torch.argmax)
// when temperature == 0, else temperature -> top-p -> multinomial.
llama_token select_token(const float* logits,
                         int n_vocab,
                         float temperature,
                         float top_p,
                         std::mt19937* rng);
// Generation loop. Stops AFTER emitting a stop token (python's generate
// keeps it in output_ids, so completion_tokens counts it).
std::vector<llama_token> generate(llama_context* ctx,
                                  const llama_vocab* vocab,
                                  int max_new,
                                  const std::vector<llama_token>& stop_ids,
                                  float temperature,
                                  float top_p,
                                  int64_t* pos_cursor,
                                  double* decode_ms);

// ---- official-style drawing ----

// ImageDraw.rectangle(xy, outline="red", width=3) semantics.
void draw_rect_pil(std::vector<uint8_t>* rgb,
                   int w,
                   int h,
                   double fx0,
                   double fy0,
                   double fx1,
                   double fy1,
                   int width);
// Red 5x7-bitmap-font label at (fx+2, fy+2), doubled above 640 px width.
// Non-ASCII glyphs render as '?' — callers fall back to the <objN> index
// first (all_ascii), matching the official UnicodeEncodeError retry.
void draw_label_pil(std::vector<uint8_t>* rgb,
                    int w,
                    int h,
                    double fx,
                    double fy,
                    const std::string& label);
bool all_ascii(const std::string& s);

// One resolved result box (answer parsing output).
struct ResultBox {
    std::string index, label;
    BoxD box;
};

}  // namespace vlx
