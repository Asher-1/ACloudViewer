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

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ocr_rec_crop.hpp"
#include "tests/common/test_macros.hpp"

static int failures = 0;

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

int main(int argc, char** argv) {
    {
        // TextDetector's first-index tie ordering on the real val_img_421
        // diamond repeats a corner. Pinned source OpenCV produces a 36x36
        // crop filled with the top-left pixel for this singular transform.
        const float box[4][2] = {
                {237, 267}, {219, 286}, {255, 285}, {219, 286}};
        int width = 0, height = 0;
        aicore::ocr::rotateCropSize(box, &width, &height);
        AICORE_CHECK(width == 36 && height == 36);
        aicore::img::Image image;
        image.width = 1280;
        image.height = 720;
        image.channels = 3;
        image.rgb.assign(static_cast<size_t>(image.width) * image.height * 3,
                         17);
        image.rgb[0] = 29;
        image.rgb[1] = 77;
        image.rgb[2] = 163;
        std::vector<uint8_t> crop(static_cast<size_t>(width) * height * 3);
        AICORE_CHECK(aicore::ocr::warpPerspectiveCubicReplicate(
                image, box, width, height, crop.data()));
        bool equal = true;
        for (size_t i = 0; i < crop.size(); ++i)
            equal = equal && crop[i] == image.rgb[i % 3];
        AICORE_CHECK(equal);
    }
#ifdef AICORE_OCR_REC_CROP_GOLDEN
    const char* golden = argc > 1 ? argv[1] : AICORE_OCR_REC_CROP_GOLDEN;
#else
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s rec_crop_golden.txt\n", argv[0]);
        return 2;
    }
    const char* golden = argv[1];
#endif
    std::ifstream in(golden);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", golden);
        return 2;
    }
    std::string line;
    int cases = 0, boxMismatch = 0, sizeMismatch = 0, cropMismatch = 0;
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
        AICORE_CHECK(static_cast<bool>(row));
        if (!row) break;
        ++cases;

        float box[4][2];
        aicore::ocr::minAreaRectBox(
                reinterpret_cast<const int(*)[2]>(pts.data()), n, box);
        bool boxOk = true;
        for (int i = 0; i < 4; ++i)
            boxOk = boxOk && box[i][0] == want[i][0] && box[i][1] == want[i][1];
        if (!boxOk && boxMismatch++ < 5)
            std::fprintf(stderr,
                         "box mismatch (row %d): got %.9g %.9g %.9g %.9g "
                         "%.9g %.9g %.9g %.9g\n",
                         cases, box[0][0], box[0][1], box[1][0], box[1][1],
                         box[2][0], box[2][1], box[3][0], box[3][1]);

        // The crop from the golden box isolates the warp from the re-fit.
        int gw = 0, gh = 0;
        aicore::ocr::rotateCropSize(want, &gw, &gh);
        if ((gw != cw || gh != ch) && sizeMismatch++ < 5)
            std::fprintf(stderr, "size mismatch (row %d): %dx%d vs %dx%d\n",
                         cases, gw, gh, cw, ch);
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
        if ((!warped || hashHex != hex) && cropMismatch++ < 5)
            std::fprintf(stderr, "crop mismatch (row %d, %dx%d): %s vs %s\n",
                         cases, cw, ch, hex, hashHex.c_str());
    }
    AICORE_CHECK(cases >= 720);
    AICORE_CHECK(boxMismatch == 0);
    AICORE_CHECK(sizeMismatch == 0);
    AICORE_CHECK(cropMismatch == 0);
    std::printf(
            "rec crop parity: %d cases, %d box / %d size / %d crop "
            "mismatches\n",
            cases, boxMismatch, sizeMismatch, cropMismatch);
    if (failures) {
        std::fprintf(stderr, "test_ocr_rec_crop: %d failure(s)\n", failures);
        return 1;
    }
    return 0;
}
