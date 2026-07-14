#pragma once
#include "image_io.hpp"
#include "model_loader.hpp"
#include <vector>
#include <cstdint>
namespace da {
struct Preprocessed {
    int H=0, W=0; std::vector<float> chw;   // [3,H,W] f32, C-major
    // Original image size and the cumulative orig->processed scale factors.
    // Intrinsics scale as: fx,cx *= scale_w ; fy,cy *= scale_h (DA3 _resize_ixt).
    int orig_w=0, orig_h=0;
    float scale_w=1.f, scale_h=1.f;
};

// Legacy path: floor each dim to a multiple of patch_size (bilinear) + ImageNet
// normalize. A no-op resize when the image is already divisible by patch_size.
// Kept for the 224 fixture and the engine parity gates.
bool preprocess(const Image& img, const Config& cfg, Preprocessed& out);

// Real DA3 InputProcessor policy (process_res_method="upper_bound_resize"):
//   1) resize longest side to cfg.img_resize_target (cv2 INTER_CUBIC if upscaling
//      else INTER_AREA), rounding new dims with Python round-half-to-even,
//   2) round each dim to the nearest multiple of patch_size via a second resize
//      (INTER_CUBIC if that step upscales else INTER_AREA),
//   3) ImageNet normalize.
// Each resize step quantizes to uint8 (matching cv2 -> PIL.fromarray -> ToTensor).
// If `rgb_u8_out` is non-null it receives the resized, BEFORE-normalize HWC RGB
// uint8 pixels at the processed (out.W,out.H) — the per-frame colors the .glb /
// COLMAP exporters need, guaranteed identical to the model-input resize.
bool preprocess_real(const Image& img, const Config& cfg, Preprocessed& out,
                     std::vector<uint8_t>* rgb_u8_out = nullptr);

// cv2-faithful uint8 resizers (HWC RGB uint8 -> HWC RGB uint8).
Image resize_cubic(const Image& src, int dw, int dh);  // INTER_CUBIC, a=-0.75
Image resize_area (const Image& src, int dw, int dh);  // INTER_AREA (downscale)
}
