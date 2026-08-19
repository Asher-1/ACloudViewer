// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// rfdetr_preprocess white-box tests (no GGUF, no image files required).
//
// Pins the RF-DETR 1.9 antialias-free bilinear resize convention
// (rfdetr.preprocess.resize_mode = "bilinear_no_antialias"): plain bilinear
// with align_corners=false (half-pixel source centers), edge clamping, planar
// channel layout, and no intermediate uint8 rounding. Expected values are
// hand-derived from the half-pixel formula, never read back from the
// implementation. Ported from the upstream rf-detr.cpp test suite (commit
// 828cb17, "pin the 1.9 resize convention and the legacy default in tests").
//
// The legacy path (Qt SmoothTransformation) has no hand-derivable output, so
// it is only sanity-checked (shape + non-degenerate values).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/test_macros.hpp"
#include "image_io.hpp"

static int failures = 0;

namespace {

// 1e-6 absolute tolerance on normalized [0, 1] values; the implementation
// computes in double so the drift is far below this.
void check_near(float got, float want, int line) {
    if (std::fabs(got - want) > 1e-6f) {
        std::fprintf(stderr, "FAIL %s:%d: got %.9f, want %.9f\n", __FILE__,
                     line, (double)got, (double)want);
        ++failures;
    }
}

}  // namespace

int main() {
    /* RF-DETR 1.9 preprocessing: float bilinear, align_corners=false,
     * antialias=false, and no intermediate uint8 rounding.
     *
     * This 2x2 -> 3x3 case pins the half-pixel offset, the edge clamping and
     * the planar channel layout, but it does NOT distinguish
     * align_corners=false from align_corners=true: at 2 -> 3 both conventions
     * map the three output columns to source x = 0, 0.5, 1, so every value
     * below is the same either way. The 2x2 -> 4x4 case that follows is the
     * one that separates them. */
    {
        const uint8_t rgb[] = {
                0, 10, 20, 100, 110, 120, 200, 210, 220, 255, 250, 245,
        };
        rfdetr_status st_ = RFDETR_ERR_IO;
        rfdetr_image* img_ = rfdetr_image_from_rgb_buffer(rgb, 2, 2, &st_);
        AICORE_CHECK(img_ != nullptr);
        AICORE_CHECK(st_ == RFDETR_OK);

        const float mean0[3] = {0.0f, 0.0f, 0.0f};
        const float std1[3] = {1.0f, 1.0f, 1.0f};
        float* data = nullptr;
        int w = 0;
        int h = 0;
        rfdetr_status pp_st = rfdetr_preprocess(img_, 3, 3, mean0, std1,
                                                /*bilinear_no_antialias*/ true,
                                                &data, &w, &h);
        AICORE_CHECK(pp_st == RFDETR_OK);
        AICORE_CHECK(w == 3);
        AICORE_CHECK(h == 3);

        /* Upscaling 2x2 -> 3x3 with half-pixel centers puts the output
         * center exactly between all four source pixels, so it is their
         * unweighted mean. Corners land outside the source centers and clamp
         * to the nearest source pixel. Channels are planar (3x3 = 9 floats
         * each), so the centers are at offsets 4, 13, 22. */
        check_near(data[4], 138.75f / 255.0f, __LINE__);
        check_near(data[13], 145.00f / 255.0f, __LINE__);
        check_near(data[22], 151.25f / 255.0f, __LINE__);
        /* Top-middle: mean of the two top pixels, red channel. */
        check_near(data[1], 50.00f / 255.0f, __LINE__);
        /* Bottom-right corner clamps to source (1,1), red channel = 255. */
        check_near(data[8], 255.00f / 255.0f, __LINE__);

        std::free(data);
        rfdetr_image_free(img_);
    }

    /* 2x2 -> 4x4: the case where align_corners=false and align_corners=true
     * actually diverge, so a mix-up cannot pass unnoticed.
     *
     * Correct (align_corners=false, half-pixel centers): scale = 2/4 = 0.5 and
     * src = (i + 0.5) * 0.5 - 0.5, giving src = -0.25, 0.25, 0.75, 1.25 for
     * i = 0..3. The outer two land outside the source pixel centers and clamp,
     * so the sampled positions are effectively 0, 0.25, 0.75, 1.
     * Wrong (align_corners=true): src = i * (2-1)/(4-1) = 0, 1/3, 2/3, 1.
     *
     * Source pixels, (x, y) -> (R, G, B):
     *   (0,0)=(0,10,20)     (1,0)=(100,110,120)
     *   (0,1)=(200,210,220) (1,1)=(255,250,245)
     * Red row 0 is [0, 100], red row 1 is [200, 255]. Along x the weights are
     * clamp / 0.25 / 0.75 / clamp, so red resolves to:
     *   src row 0:            [  0,  25.0000,  75.0000, 100  ]
     *   0.75*row0+0.25*row1:  [ 50,  72.1875, 116.5625, 138.75]
     *   0.25*row0+0.75*row1:  [150, 166.5625, 199.6875, 216.25]
     *   src row 1:            [200, 213.7500, 241.2500, 255  ]
     * Channels are planar with a 4x4 = 16 float stride, so the index of
     * (c, y, x) is c*16 + y*4 + x. */
    {
        const uint8_t rgb[] = {
                0, 10, 20, 100, 110, 120, 200, 210, 220, 255, 250, 245,
        };
        rfdetr_status st_ = RFDETR_ERR_IO;
        rfdetr_image* img_ = rfdetr_image_from_rgb_buffer(rgb, 2, 2, &st_);
        AICORE_CHECK(img_ != nullptr);
        AICORE_CHECK(st_ == RFDETR_OK);

        const float mean0[3] = {0.0f, 0.0f, 0.0f};
        const float std1[3] = {1.0f, 1.0f, 1.0f};
        float* data = nullptr;
        int w = 0;
        int h = 0;
        rfdetr_status pp_st = rfdetr_preprocess(img_, 4, 4, mean0, std1,
                                                /*bilinear_no_antialias*/ true,
                                                &data, &w, &h);
        AICORE_CHECK(pp_st == RFDETR_OK);
        AICORE_CHECK(w == 4);
        AICORE_CHECK(h == 4);

        /* The four discriminating values. Each differs substantially under
         * align_corners=true, so this is what makes the convention testable:
         *   (0,1) red: 25.0000 here vs 33.3333 if align_corners were true
         *   (0,2) red: 75.0000 here vs 66.6667
         *   (1,0) red: 50.0000 here vs 66.6667
         *   (1,1) red: 72.1875 here vs 95.0000 */
        check_near(data[1], 25.0000f / 255.0f, __LINE__);
        check_near(data[2], 75.0000f / 255.0f, __LINE__);
        check_near(data[4], 50.0000f / 255.0f, __LINE__);
        check_near(data[5], 72.1875f / 255.0f, __LINE__);
        /* Third row, second column: exercises the other vertical weight. */
        check_near(data[9], 166.5625f / 255.0f, __LINE__);
        /* Opposite corners clamp to the nearest source pixel (same under
         * either convention, so these guard clamping, not the convention). */
        check_near(data[0], 0.0000f / 255.0f, __LINE__);
        check_near(data[15], 255.0000f / 255.0f, __LINE__);
        /* Green plane (c=1) and blue plane (c=2) confirm the 16-float stride
         * rather than an interleaved or transposed layout. */
        check_near(data[17], 35.0000f / 255.0f, __LINE__);
        check_near(data[37], 90.3125f / 255.0f, __LINE__);

        std::free(data);
        rfdetr_image_free(img_);
    }

    /* The borrowed-buffer fast path (rfdetr_image_borrow_rgb, no pixel copy)
     * must produce byte-identical output to the owning path above: the
     * bilinear branch reads through rfdetr_image_rgb_data, which is the only
     * accessor that knows about borrowed_rgb. */
    {
        const uint8_t rgb[] = {
                0, 10, 20, 100, 110, 120, 200, 210, 220, 255, 250, 245,
        };
        rfdetr_status st_ = RFDETR_ERR_IO;
        rfdetr_image* img_ = rfdetr_image_borrow_rgb(rgb, 2, 2, &st_);
        AICORE_CHECK(img_ != nullptr);
        AICORE_CHECK(st_ == RFDETR_OK);

        const float mean0[3] = {0.0f, 0.0f, 0.0f};
        const float std1[3] = {1.0f, 1.0f, 1.0f};
        float* data = nullptr;
        int w = 0;
        int h = 0;
        rfdetr_status pp_st = rfdetr_preprocess(img_, 4, 4, mean0, std1,
                                                /*bilinear_no_antialias*/ true,
                                                &data, &w, &h);
        AICORE_CHECK(pp_st == RFDETR_OK);
        check_near(data[5], 72.1875f / 255.0f, __LINE__);
        check_near(data[9], 166.5625f / 255.0f, __LINE__);
        check_near(data[37], 90.3125f / 255.0f, __LINE__);

        std::free(data);
        rfdetr_image_free(img_);
    }

    /* Legacy path (GGUFs without the resize_mode key): Qt SmoothTransformation
     * has no hand-derivable output, so only sanity-check shape and that the
     * normalization did not kill the signal. */
    {
        const uint8_t rgb[] = {
                0, 10, 20, 100, 110, 120, 200, 210, 220, 255, 250, 245,
        };
        rfdetr_status st_ = RFDETR_ERR_IO;
        rfdetr_image* img_ = rfdetr_image_from_rgb_buffer(rgb, 2, 2, &st_);
        AICORE_CHECK(img_ != nullptr);

        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std_[3] = {0.229f, 0.224f, 0.225f};
        float* data = nullptr;
        int w = 0;
        int h = 0;
        rfdetr_status pp_st = rfdetr_preprocess(img_, 56, 56, mean, std_,
                                                /*bilinear_no_antialias*/
                                                false, &data, &w, &h);
        AICORE_CHECK(pp_st == RFDETR_OK);
        AICORE_CHECK(w == 56);
        AICORE_CHECK(h == 56);
        AICORE_CHECK(data != nullptr);

        bool all_zero = true;
        for (int i = 0; i < 56 * 56 * 3 && all_zero; ++i) {
            if (data[i] != 0.0f) all_zero = false;
        }
        AICORE_CHECK(!all_zero);

        std::free(data);
        rfdetr_image_free(img_);
    }

    if (failures == 0) {
        std::printf("[rfdetr] preprocess pin test passed\n");
    }
    return failures;
}
