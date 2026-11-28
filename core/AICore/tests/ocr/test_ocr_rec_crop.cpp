// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Rec crop parity against upstream (no model, no backend). Every golden row
// was produced by PaddleX CropByPolys("quad") get_minarea_rect_crop:
// cv2.minAreaRect + cv2.boxPoints re-fit, int() float32 edge norms and
// cv2.warpPerspective(INTER_CUBIC, BORDER_REPLICATE) (scripts/
// gen_ocr_rec_crop_golden.py). Box, size and crop bytes must match exactly.

#include <gtest/gtest.h>

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tasks/ocr/ocr_rec_crop.hpp"

#ifdef AICORE_OCR_REC_CROP_GOLDEN
static const char* golden_path = AICORE_OCR_REC_CROP_GOLDEN;
#else
static const char* golden_path = nullptr;
#endif

namespace {

// gen_ocr_rec_crop_golden.py hash_image, HxWx3 interleaved.
std::vector<uint8_t> hashImage(int w, int h, uint64_t salt) {
    std::vector<uint8_t> img(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                uint64_t v = ((uint64_t)x * 73856093u) ^
                             ((uint64_t)y * 19349663u) ^
                             ((uint64_t)c * 83492791u) ^ salt;
                v = (v * 2654435761u) & 0xFFFFFFFFu;
                img[(static_cast<size_t>(y) * w + x) * 3 + c] =
                        static_cast<uint8_t>((v >> 13) & 255);
            }
    return img;
}

uint64_t fnv1a64(const std::vector<uint8_t>& data) {
    uint64_t value = 0xcbf29ce484222325ull;
    for (const uint8_t byte : data) value = (value ^ byte) * 0x100000001b3ull;
    return value;
}

}  // namespace

TEST(OcrRecCrop, SingularTransformReplicatesTopLeftPixel) {
    // TextDetector's first-index tie ordering on the real val_img_421
    // diamond repeats a corner. Pinned source OpenCV produces a 36x36
    // crop filled with the top-left pixel for this singular transform.
    const float box[4][2] = {{237, 267}, {219, 286}, {255, 285}, {219, 286}};
    int width = 0, height = 0;
    aicore::ocr::rotateCropSize(box, &width, &height);
    ASSERT_EQ(width, 36);
    ASSERT_EQ(height, 36);
    aicore::img::Image image;
    image.width = 1280;
    image.height = 720;
    image.channels = 3;
    image.rgb.assign(static_cast<size_t>(image.width) * image.height * 3, 17);
    image.rgb[0] = 29;
    image.rgb[1] = 77;
    image.rgb[2] = 163;
    std::vector<uint8_t> crop(static_cast<size_t>(width) * height * 3);
    ASSERT_TRUE(aicore::ocr::warpPerspectiveCubicReplicate(
            image, box, width, height, crop.data()));
    bool equal = true;
    for (size_t i = 0; i < crop.size(); ++i)
        equal = equal && crop[i] == image.rgb[i % 3];
    EXPECT_TRUE(equal);
}

TEST(OcrRecCrop, MatchesUpstreamBoxesSizesAndCropBytes) {
    ASSERT_NE(golden_path, nullptr) << "Supply the golden fixture path as the "
                                       "first positional argument";
    std::ifstream in(golden_path);
    ASSERT_TRUE(in.is_open()) << "Cannot open " << golden_path;
    std::string line;
    int cases = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        int iw = 0, ih = 0, n = 0;
        uint64_t salt = 0;
        row >> iw >> ih >> salt >> n;
        std::vector<int> pts(static_cast<size_t>(n) * 2);
        for (int& v : pts) row >> v;
        std::string bar;
        float want[4][2];
        row >> bar;
        for (auto& corner : want) row >> corner[0] >> corner[1];
        int cw = 0, ch = 0;
        std::string hashHex;
        row >> bar >> cw >> ch >> hashHex;
        ASSERT_TRUE(static_cast<bool>(row)) << line;
        ++cases;
        SCOPED_TRACE(testing::Message() << "Golden row " << cases);

        float box[4][2];
        aicore::ocr::minAreaRectBox(
                reinterpret_cast<const int(*)[2]>(pts.data()), n, box);
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(box[i][0], want[i][0]) << "Corner " << i << " x";
            EXPECT_EQ(box[i][1], want[i][1]) << "Corner " << i << " y";
        }

        // The crop from the golden box isolates the warp from the re-fit.
        int gw = 0, gh = 0;
        aicore::ocr::rotateCropSize(want, &gw, &gh);
        EXPECT_EQ(gw, cw);
        EXPECT_EQ(gh, ch);
        if (gw != cw || gh != ch) continue;

        aicore::img::Image src;
        src.width = iw;
        src.height = ih;
        src.channels = 3;
        src.rgb = hashImage(iw, ih, salt);
        std::vector<uint8_t> crop(static_cast<size_t>(cw) * ch * 3);
        const bool warped = aicore::ocr::warpPerspectiveCubicReplicate(
                src, want, cw, ch, crop.data());
        char hex[17];
        std::snprintf(hex, sizeof(hex), "%016" PRIx64, fnv1a64(crop));
        EXPECT_TRUE(warped);
        EXPECT_EQ(hashHex, hex);
    }
    EXPECT_GE(cases, 720);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    if (argc > 1) golden_path = argv[1];
    return RUN_ALL_TESTS();
}
