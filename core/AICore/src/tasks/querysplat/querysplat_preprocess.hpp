// Preprocessing for the QuerySplat chain (pure functions, no file I/O).
// Ported from upstream cpp_ggml image_io.cpp: the official 512 preprocessing
// (torchvision ImageTransform(max_crop=True): center square crop of side
// min(h,w), then Resize) and the F.interpolate(mode="bilinear",
// align_corners=False, antialias=True) resize it relies on. File decoding
// stays at the caller (plugin Qt / CLI): the typed AICore input is an
// already-decoded interleaved RGB8 image per view.
#pragma once

#include <cstdint>
#include <vector>

namespace qs {

// (sh,sw,3) interleaved -> (H,W,3) interleaved, F.interpolate bilinear with
// antialias=True: plain bilinear (ATen upsample_bilinear2d accumulation order)
// when neither axis shrinks, the separable antialias triangle filter otherwise.
void resize_rgb_hwc(const float* src, int sh, int sw, int H, int W, float* dst);

// The same resize on (n,3,h,w) planes -> (n,3,H,W).
std::vector<float> resize_planes(const float* src, int n, int h, int w, int H,
                                 int W);

// One decoded RGB8 view -> the official preprocessing chain:
// center square crop of side min(h,w) -> normalize to [0,1] -> antialiased
// resize to (H,W) -> (3,H,W) f32 planes, torch (3,H,W) C-order. This is the
// byte-exact composition load_images_as_frames applies after decoding
// (crop extraction and HWC->CHW included), verified against the official
// python ImageTransform to 2.4e-7 on the 4135->512 geometry.
std::vector<float> preprocess_rgb8_view(const uint8_t* rgb8, int w, int h,
                                        int H, int W);

// Every supported aicore view format routes through an RGB8 de-interleave
// first; GRAY replicates channels. Returns false on unsupported formats or
// non-positive dimensions.
bool preprocess_view_bytes(const uint8_t* data, int w, int h,
                           int channels_per_pixel, bool bgr_order,
                           int H, int W, std::vector<float>* out_frames);

}  // namespace qs
