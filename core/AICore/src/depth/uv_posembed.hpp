#pragma once
#include <vector>
namespace aicore {
namespace depth {
// Returns the UV sinusoidal positional embedding flattened in (H, W, C) row-major order
// (matching numpy reshape(-1) of a (ph, pw, C) tensor): element (y,x,c) at (y*pw + x)*C + c.
// This is the RAW embed (NOT multiplied by the 0.1 ratio). aspect = W/H.
std::vector<float> uv_pos_embed(int pw, int ph, int C, float aspect, float omega0 = 100.f);
} // namespace depth
} // namespace aicore
