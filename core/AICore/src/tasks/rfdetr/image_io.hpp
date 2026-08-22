#pragma once


#include "tasks/rfdetr/rfdetr.h"


#include <cstdint>
#include <vector>

struct rfdetr_image {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<uint8_t> rgb;  /* HWC, row-major, 0..255 */
    /* Internal fast path: non-owning pointer to an external RGB888 buffer.
     * When set, `rgb` is left empty and every pixel read goes through this
     * pointer; the caller must keep the buffer alive for the image's whole
     * lifetime. Owned images (load_file / from_rgb_buffer) never set it. */
    const uint8_t* borrowed_rgb = nullptr;
};

#ifdef __cplusplus
extern "C" {
#endif

const uint8_t* rfdetr_image_rgb_data(const rfdetr_image* img);

/* Preprocess an image for model input:
 *   1. Resize to (target_w, target_h) — the path is selected by
 *      `bilinear_no_antialias` (from the GGUF's rfdetr.preprocess.resize_mode
 *      metadata):
 *        - true:  RF-DETR 1.9 antialias-free bilinear, matching torchvision
 *                 F.resize(..., antialias=False): plain bilinear with
 *                 half-pixel source centers, no prefilter, coordinates in
 *                 double precision, and no round-trip through uint8.
 *        - false: legacy Qt SmoothTransformation resize, preserved so GGUFs
 *                 converted before the resize_mode key existed keep their
 *                 exact outputs.
 *   2. Convert uint8 RGB -> float32 in [0, 1]
 *   3. Apply ImageNet normalization: (pixel - mean) / std (per channel)
 *   4. Output in (W, H, 3, 1) ggml layout, NCHW row-major equivalent
 *
 * The output buffer is allocated by the function; caller frees with std::free.
 *
 * Returns RFDETR_OK and fills *out_data + *out_w + *out_h on success. */
rfdetr_status rfdetr_preprocess(const rfdetr_image* img,
                                int target_w, int target_h,
                                const float mean[3], const float std_[3],
                                bool bilinear_no_antialias,
                                float** out_data, int* out_w, int* out_h);

/* Write a single-channel uint8 (grayscale) buffer as a PNG. Returns
 * RFDETR_OK on success. `data` is row-major, size = width * height bytes. */
rfdetr_status rfdetr_write_gray_png(const char* path,
                                    const uint8_t* data,
                                    int width, int height);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
/* C++-only helper: PNG-encode a grayscale buffer into an in-memory vector.
 * Same input layout as rfdetr_write_gray_png (row-major, 1 byte per pixel).
 * Used by the flat C-API accessor for serving masks to LocalAI without
 * hitting disk. Returns true on success, false on encoding failure. */
bool rfdetr_encode_gray_png(const uint8_t* data, int width, int height,
                            std::vector<uint8_t>& out);

/* C++-only helper: wrap an external RGB888 buffer into an rfdetr_image
 * WITHOUT copying the pixels (borrowed mode — see borrowed_rgb above).
 * Intended for internal callers whose input buffer outlives the call (e.g.
 * the flat RGB C-API path); the public from_rgb_buffer keeps its owning
 * semantics. Returns nullptr on invalid arguments / OOM. */
rfdetr_image* rfdetr_image_borrow_rgb(const uint8_t* rgb, int width,
                                      int height, rfdetr_status* out_status);
#endif
