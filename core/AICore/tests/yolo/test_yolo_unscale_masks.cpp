// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Whitebox test for yolo::unscale_masks(): remapping instance masks from
// the letterbox-canvas space back to the source image space. No GGUF assets
// needed — pure geometry. Catches regressions in the canvas -> source
// mapping (padding offset / scale) that would silently misplace the segment
// tint on the original image.

#include <cstdio>
#include <vector>

#include "tasks/yolo/yolo_image.hpp"
#include "tasks/yolo/yolo_postprocess.hpp"

namespace {

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                         \
            return 1;                                                    \
        }                                                                \
    } while (0)

// Source pixel (x, y) -> canvas window coordinate, mirroring the sampling
// convention of unscale_masks (nearest, floor((p + 0.5) * scale - 0.5) + pad
// minus the window origin).
int canvasCoord(int p, float scale, int pad, int winOrigin) {
    return static_cast<int>((p + 0.5f) * scale - 0.5f) + pad - winOrigin;
}

int runCase1() {
    // Exact 2x upscale, no padding: canvas 640x640 over a 320x320 source.
    // Canvas window (100..140, 50..90) must land on source (50..70, 25..45).
    yolo::SegMask mask;
    mask.x = 100;
    mask.y = 50;
    mask.w = 40;
    mask.h = 40;
    mask.bits.assign(static_cast<size_t>(mask.w) * mask.h, 1);
    std::vector<yolo::SegMask> masks{mask};
    yolo::LetterboxInfo info{2.0f, 0, 0, 320, 320, 640, 640};
    yolo::unscale_masks(masks, info, 320, 320);

    CHECK(masks.size() == 1);
    CHECK(masks[0].w == 320 && masks[0].h == 320);
    CHECK(masks[0].x == 0 && masks[0].y == 0);
    CHECK(masks[0].bits.size() == 320 * 320);
    CHECK(masks[0].bits[25 * 320 + 50] == 1);  // window top-left
    CHECK(masks[0].bits[44 * 320 + 69] == 1);  // window bottom-right
    CHECK(masks[0].bits[24 * 320 + 50] == 0);  // one row above the window
    CHECK(masks[0].bits[45 * 320 + 50] == 0);  // one row below
    CHECK(masks[0].bits[25 * 320 + 49] == 0);  // one column left
    CHECK(masks[0].bits[25 * 320 + 70] == 0);  // one column right
    CHECK(masks[0].bits[0] == 0);
    CHECK(masks[0].bits[319 * 320 + 319] == 0);
    return 0;
}

int runCase2() {
    // Letterbox with vertical padding: source 640x427, scale 1.2, canvas
    // 768x768 with pad_h = 128 (pad_w = 0). A canvas window fully inside the
    // padded area must land at the same offset in the source image.
    yolo::SegMask mask;
    mask.x = 0;
    mask.y = 128 + 50;  // 50 canvas rows below the padded top edge
    mask.w = 200;
    mask.h = 200;
    mask.bits.assign(static_cast<size_t>(mask.w) * mask.h, 1);
    std::vector<yolo::SegMask> masks{mask};
    yolo::LetterboxInfo info{1.2f, 0, 128, 768, 512, 768, 768};
    yolo::unscale_masks(masks, info, 640, 427);

    CHECK(masks[0].w == 640 && masks[0].h == 427);
    CHECK(masks[0].x == 0 && masks[0].y == 0);
    // Window interior must be set, boundaries follow the nearest-neighbor
    // convention of unscale_masks itself. The window sits 50 canvas rows
    // below the padded top edge, i.e. source rows [42, 209) (50 / 1.2).
    const int yTop = [&]() {
        for (int y = 0; y < 427; ++y) {
            if (canvasCoord(y, 1.2f, 128, mask.y) >= 0) return y;
        }
        return 427;
    }();
    const int yBottom = [&]() {
        for (int y = 426; y >= 0; --y) {
            if (canvasCoord(y, 1.2f, 128, mask.y) < mask.h) return y;
        }
        return -1;
    }();
    const int xRight = [&]() {
        for (int x = 639; x >= 0; --x) {
            if (canvasCoord(x, 1.2f, 0, mask.x) < mask.w) return x;
        }
        return -1;
    }();
    CHECK(yTop == 42);
    CHECK(yBottom == 208);
    CHECK(xRight == 166);
    const int yMid = (yTop + yBottom) / 2;
    const int xMid = 80;
    CHECK(masks[0].bits[yMid * 640 + xMid] == 1);     // interior
    CHECK(masks[0].bits[yTop * 640 + xMid] == 1);     // top edge inside
    CHECK(masks[0].bits[yBottom * 640 + xMid] == 1);  // bottom edge inside
    if (yTop > 0) CHECK(masks[0].bits[(yTop - 1) * 640 + xMid] == 0);
    if (yBottom < 426) CHECK(masks[0].bits[(yBottom + 1) * 640 + xMid] == 0);
    CHECK(masks[0].bits[yMid * 640 + 0] == 1);  // x starts at the left
    CHECK(masks[0].bits[yMid * 640 + xRight] == 1);
    if (xRight < 639) CHECK(masks[0].bits[yMid * 640 + xRight + 1] == 0);
    CHECK(masks[0].bits[0] == 0);                // padded top rows -> nothing
    CHECK(masks[0].bits[426 * 640 + 639] == 0);  // outside the window
    return 0;
}

int runCase3() {
    // Degenerate inputs must not crash and leave the mask untouched.
    yolo::SegMask empty;  // w = h = 0
    std::vector<yolo::SegMask> masks{empty};
    yolo::LetterboxInfo info{1.0f, 0, 0, 8, 8, 8, 8};
    yolo::unscale_masks(masks, info, 8, 8);
    CHECK(masks[0].w == 0 && masks[0].h == 0);
    yolo::unscale_masks(masks, info, 0, 8);  // invalid source dims
    CHECK(masks[0].w == 0 && masks[0].h == 0);
    return 0;
}

int runStrideAwarePreprocessCase() {
    constexpr int w = 3;
    constexpr int h = 2;
    const uint8_t packed[w * h * 3] = {10, 20, 30, 40, 50, 60, 70, 80, 90,
                                       11, 21, 31, 41, 51, 61, 71, 81, 91};
    const uint8_t padded[h * 12] = {10, 20, 30, 40, 50, 60, 70, 80,
                                    90, 1,  2,  3,  11, 21, 31, 41,
                                    51, 61, 71, 81, 91, 4,  5,  6};
    const uint8_t rgba[w * h * 4] = {10, 20, 30, 255, 40, 50, 60, 0,
                                     70, 80, 90, 127, 11, 21, 31, 1,
                                     41, 51, 61, 2,   71, 81, 91, 3};
    const uint8_t bgr[w * h * 3] = {30, 20, 10, 60, 50, 40, 90, 80, 70,
                                    31, 21, 11, 61, 51, 41, 91, 81, 71};
    const uint8_t bgra[w * h * 4] = {30, 20, 10, 255, 60, 50, 40, 0,
                                     90, 80, 70, 127, 31, 21, 11, 1,
                                     61, 51, 41, 2,   91, 81, 71, 3};

    yolo::LetterboxInfo packedInfo, paddedInfo, rgbaInfo, bgrInfo, bgraInfo;
    std::vector<float> packedOut, paddedOut, rgbaOut, bgrOut, bgraOut;
    yolo::letterbox_image(yolo::Image{w, h, packed, w * 3, 3}, 32, packedInfo,
                          packedOut);
    yolo::letterbox_image(yolo::Image{w, h, padded, 12, 3}, 32, paddedInfo,
                          paddedOut);
    yolo::letterbox_image(yolo::Image{w, h, rgba, w * 4, 4}, 32, rgbaInfo,
                          rgbaOut);
    yolo::letterbox_image(yolo::Image{w, h, bgr, w * 3, 3, true}, 32, bgrInfo,
                          bgrOut);
    yolo::letterbox_image(yolo::Image{w, h, bgra, w * 4, 4, true}, 32, bgraInfo,
                          bgraOut);
    CHECK(packedOut == paddedOut);
    CHECK(packedOut == rgbaOut);
    CHECK(packedOut == bgrOut);
    CHECK(packedOut == bgraOut);

    std::vector<float> packedCls, paddedCls, rgbaCls, bgrCls, bgraCls;
    yolo::classify_preprocess(yolo::Image{w, h, packed, w * 3, 3}, 2,
                              packedCls);
    yolo::classify_preprocess(yolo::Image{w, h, padded, 12, 3}, 2, paddedCls);
    yolo::classify_preprocess(yolo::Image{w, h, rgba, w * 4, 4}, 2, rgbaCls);
    yolo::classify_preprocess(yolo::Image{w, h, bgr, w * 3, 3, true}, 2,
                              bgrCls);
    yolo::classify_preprocess(yolo::Image{w, h, bgra, w * 4, 4, true}, 2,
                              bgraCls);
    CHECK(packedCls == paddedCls);
    CHECK(packedCls == rgbaCls);
    CHECK(packedCls == bgrCls);
    CHECK(packedCls == bgraCls);

    const uint8_t gray[w * h] = {10, 40, 70, 11, 41, 71};
    const uint8_t grayRgb[w * h * 3] = {10, 10, 10, 40, 40, 40, 70, 70, 70,
                                        11, 11, 11, 41, 41, 41, 71, 71, 71};
    std::vector<float> grayOut, grayRgbOut;
    yolo::letterbox_image(yolo::Image{w, h, gray, w, 1}, 32, packedInfo,
                          grayOut);
    yolo::letterbox_image(yolo::Image{w, h, grayRgb, w * 3, 3}, 32, paddedInfo,
                          grayRgbOut);
    CHECK(grayOut == grayRgbOut);
    return 0;
}

}  // namespace

int main() {
    int rc = runCase1();
    if (rc != 0) return rc;
    rc = runCase2();
    if (rc != 0) return rc;
    rc = runCase3();
    if (rc != 0) return rc;
    rc = runStrideAwarePreprocessCase();
    if (rc != 0) return rc;
    std::printf("test_yolo_unscale_masks: all checks passed\n");
    return 0;
}
