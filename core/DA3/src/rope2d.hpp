#pragma once
#include "ggml.h"
#include "backend.hpp"
#include <vector>
namespace da {
struct RopeTables { int head_dim=0, tokens=0; std::vector<float> cos, sin; };
// pos_yx: flat [tokens*2] as (y,x) per token (already +1/special-adjusted by caller).
RopeTables build_rope_tables(const std::vector<float>& pos_yx, int tokens, int head_dim, float freq);
// Apply rope to x [head_dim, heads, tokens]; cosb/sinb are [head_dim,1,tokens] inputs.
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cosb, ggml_tensor* sinb, int head_dim);
void build_rope_inputs(ggml_context* ctx, Backend& be, GraphInputPool& pool,
                       const RopeTables& rt, ggml_tensor*& cosb, ggml_tensor*& sinb);
}
