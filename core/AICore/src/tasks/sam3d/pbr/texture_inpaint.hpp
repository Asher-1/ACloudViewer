// Native counterpart of the final cv2.INPAINT_TELEA atlas repair stage.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "asset_io.hpp"

namespace sam3d {

// `hole_mask` is one byte per texel: non-zero values identify the texels to
// reconstruct. This is the same polarity as `1 - rasterize(...)["mask"]` in
// the official optimized texture-bake path.
bool inpaint_texture_telea(RgbaImage& texture, const std::vector<uint8_t>& hole_mask,
                           float radius, std::string& error);

}  // namespace sam3d
